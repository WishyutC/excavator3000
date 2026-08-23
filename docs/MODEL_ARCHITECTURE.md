# Excavator3000 DQN model architecture

## Purpose

The Excavator3000 neural network is a compact decision model for local obstacle
avoidance. It converts eight proximity measurements and two motion measurements
into one of three high-level driving decisions:

```text
10 normalized inputs -> neural network -> 3 Q-values -> selected action
```

The model decides **what the vehicle should do**. It does not directly output
motor PWM, voltage, RPM, position, or steering angle. Those physical controls
belong to the ESP32 motor-control and safety layers.

## End-to-end data flow

```text
Eight distance sensors         Webots/vehicle motion estimate
          |                          |
          v                          v
lookup-table distance          forward speed + turn rate
          |                          |
          +------------+-------------+
                       |
                       v
            normalize and clip values
                       |
                       v
             float32 tensor [1, 10]
                       |
                       v
       Dense(10,64) -> ReLU -> Dense(64,64)
                       |
                      ReLU
                       |
                       v
                Dense(64,3)
                       |
                       v
             Q-forward, Q-left, Q-right
                       |
                       v
        argmax during deployment/evaluation
                       |
                       v
              high-level motor action
```

## Input layer

### Shape and data type

| Context | Tensor shape | Type |
|---|---|---|
| Single inference | `[1, 10]` | `float32` before quantization |
| Training batch | `[batch_size, 10]` | `float32` |

The ordering is part of the model contract. It must remain identical in Webots,
model conversion, and ESP32 firmware.

### Exact input order

| Index | Feature | Normalized range | Meaning |
|---:|---|---:|---|
| 0 | Front proximity | 0 to 1 | Obstacle directly ahead |
| 1 | Back proximity | 0 to 1 | Obstacle directly behind |
| 2 | Left proximity | 0 to 1 | Obstacle at the left side |
| 3 | Right proximity | 0 to 1 | Obstacle at the right side |
| 4 | Left-front proximity | 0 to 1 | Obstacle ahead-left |
| 5 | Right-front proximity | 0 to 1 | Obstacle ahead-right |
| 6 | Left-back proximity | 0 to 1 | Obstacle behind-left |
| 7 | Right-back proximity | 0 to 1 | Obstacle behind-right |
| 8 | Signed forward speed | -1 to 1 | Reverse through forward motion |
| 9 | Signed turn rate | -1 to 1 | Rotation around the vertical axis |

The sensor device order in `robot_controller.py` exactly matches this table.

## Input preprocessing

### Sensor lookup-table conversion

Webots distance sensors produce raw device values. These values are not fed
directly into the network. For each sensor, the controller interpolates the
sensor's Webots lookup table to estimate distance in metres.

The distance is then converted to proximity:

```text
proximity = clip(1 - distance_m / 0.8, 0, 1)
```

Interpretation:

- `0.0` means clear space at or beyond the configured 0.8 m sensor range;
- `0.5` means an estimated obstacle distance of 0.4 m;
- `1.0` means an obstacle at the sensor or the minimum measurable distance.

If a distance is unavailable or non-finite, the current preprocessing returns
`0.0`. Real firmware should treat sensor faults separately in its safety layer
rather than assuming every missing measurement is safe.

### Signed forward speed

Webots supplies velocity in world coordinates. The controller projects linear
velocity onto the robot's local forward axis:

```text
forward_speed = dot(world_linear_velocity, robot_forward_axis)
normalized_speed = clip(forward_speed / 0.129, -1, 1)
```

Positive values mean forward movement and negative values mean reverse
movement. This makes the input independent of the robot's world heading.

For the real vehicle, this value should come from wheel encoders, an odometry
estimate, or another calibrated motion source. It should not be replaced by the
requested PWM percentage because commanded speed and actual speed can differ.

### Signed turn rate

The final input is angular velocity around the vertical axis:

```text
normalized_turn_rate = clip(angular_rate_rad_s / 2.0, -1, 1)
```

The ESP32 system can obtain this from an IMU gyroscope or a calibrated estimate
from wheel encoders. The sign convention used by the firmware must match the
Webots convention used during model validation.

### Inputs deliberately excluded

The policy does not receive:

- map name or layout identifier;
- world X/Y position or heading;
- spawn location;
- goal or checkpoint coordinates;
- episode number;
- reward or termination reason;
- raw Webots sensor values;
- privileged Supervisor information.

Excluding these inputs prevents direct map memorization and keeps the deployed
interface based on measurements a real vehicle can produce.

## Neural-network layers

The model is implemented by `DQNNetwork` in `dqn_network.py`.

| Layer | Input | Output | Activation | Parameters |
|---|---:|---:|---|---:|
| Dense 1 | 10 | 64 | ReLU | `10 x 64 + 64 = 704` |
| Dense 2 | 64 | 64 | ReLU | `64 x 64 + 64 = 4,160` |
| Output | 64 | 3 | Linear | `64 x 3 + 3 = 195` |
| **Total** | | | | **5,059** |

Equivalent PyTorch structure:

```text
Sequential(
  Linear(in_features=10, out_features=64)
  ReLU()
  Linear(in_features=64, out_features=64)
  ReLU()
  Linear(in_features=64, out_features=3)
)
```

There is no Softmax output. DQN outputs unbounded action values, and converting
them into probabilities would change the policy semantics.

### Memory size

The raw parameter memory is approximately:

```text
5,059 parameters x 4 bytes per float32 = 20,236 bytes
```

This is about 19.8 KiB for weights and biases alone. Runtime activation buffers,
tensor metadata, and the inference engine require additional memory. Full
integer quantization could reduce parameter storage to approximately 5 KiB,
but the exact TensorFlow Lite Micro arena requirement must be measured after
conversion.

## Output layer

### Shape

| Context | Tensor shape |
|---|---|
| Single inference | `[1, 3]` |
| Training batch | `[batch_size, 3]` |

The output vector is:

```text
[Q(forward), Q(turn_left), Q(turn_right)]
```

Each value estimates the discounted future return expected after taking that
action in the current state and continuing with the learned policy.

Example:

```text
network output = [0.42, 1.18, -0.31]
selected index = argmax(output) = 1
decision = TURN LEFT
```

Q-values are relative decision scores. A value of `1.18` does not mean 118%
confidence, 1.18 rad/s, or a 1.18 V motor command.

## Action mapping

The training policy is restricted by `training.action_ids = [0, 1, 2]`:

| Policy output index | Robot action | Left wheel ratio | Right wheel ratio |
|---:|---|---:|---:|
| 0 | Forward | 1.00 | 1.00 |
| 1 | Turn left | 0.00 | 0.80 |
| 2 | Turn right | 0.80 | 0.00 |

Wheel ratios multiply the controller's safe drive speed:

```text
drive_speed = min(Webots motor limit, configured limit) x 0.95
wheel_velocity = wheel_ratio x drive_speed
```

In the survival maps, each decision is normally held for four 32 ms physics
steps, or approximately 128 ms, unless the episode terminates first. The
race-track committed-turn macro is disabled automatically for survival training.

The real ESP32 implementation does not have to copy these raw Webots angular
velocities. The hardware team should tune three equivalent maneuvers while
preserving their meaning and response timing.

## Action selection during training and deployment

During training, epsilon-greedy exploration selects:

```text
random action, with probability epsilon
argmax(Q-values), otherwise
```

Epsilon currently decreases linearly from 1.00 to 0.05 over 1,000,000 agent
decisions. This randomness exists only to collect varied training experience.

During frozen evaluation and ESP32 deployment:

```text
action = argmax(Q-values)
```

No replay memory, reward calculation, epsilon exploration, target network,
optimizer, or backpropagation is required on the ESP32.

## Double DQN training architecture

PC training contains two networks with the same 10→64→64→3 structure:

| Network | Role |
|---|---|
| Online network | Selects actions and receives gradient updates |
| Target network | Supplies stable next-state values for Bellman targets |

For each replay transition `(state, action, reward, next_state, done)`, Double
DQN uses:

```text
next_action = argmax(online_network(next_state))
next_value  = target_network(next_state)[next_action]
target      = scaled_reward + gamma x (1 - done) x next_value
```

The selected online Q-value is trained toward this target using Huber loss
(`SmoothL1Loss`). Current training settings include:

- Adam optimizer with learning rate 0.0002;
- discount factor `gamma = 0.999`;
- reward scale 0.01 before the Bellman update;
- gradient-norm clipping at 10.0;
- target-network synchronization every 1,000 optimizer updates;
- replay batch size 64;
- replay capacity 50,000 transitions;
- learning warmup of 2,000 transitions;
- one optimizer update every four agent decisions.

The 100,000-episode run checkpoints both networks, optimizer state, counters,
and replay memory for outage-safe recovery.

## PyTorch-to-ESP32 deployment contract

The deployment model must preserve:

1. input order `[front, back, left, right, left-front, right-front,
   left-back, right-back, forward-speed, turn-rate]`;
2. the exact proximity, speed, and turn-rate normalization equations;
3. input shape `[1, 10]`;
4. output order `[forward, left, right]`;
5. linear Q-value output with `argmax` action selection;
6. equivalent action timing and meaning.

Planned conversion path:

```text
PyTorch checkpoint
      -> exportable inference network
      -> interoperable conversion representation
      -> TensorFlow Lite model
      -> integer quantization
      -> TensorFlow Lite Micro on ESP32
```

Before installation, run identical observation vectors through PyTorch and the
converted model. Compare all three outputs and, most importantly, confirm that
both models select the same `argmax` action. Quantization is acceptable only if
action agreement and held-out-map performance remain sufficiently close.

## ESP32 inference pseudocode

```text
loop:
    distances = read_and_calibrate_8_sensors()
    forward_speed = estimate_signed_forward_speed()
    turn_rate = read_signed_turn_rate()

    input[0..7] = normalize_proximity(distances, max_distance=0.8 m)
    input[8] = clip(forward_speed / 0.129 m/s, -1, 1)
    input[9] = clip(turn_rate / 2.0 rad/s, -1, 1)

    q_values = tflite_inference(input)
    action = argmax(q_values)

    if independent_emergency_sensor_check_failed():
        stop_motors()
    else:
        execute_calibrated_action(action)
```

Independent emergency protection must remain outside the neural network. The
policy may choose a poor action, receive bad sensor data, or encounter a real
situation that was not represented in simulation.

## Source of truth

Architecture and behavior are defined by:

- `observation.py` for normalization and input construction;
- `dqn_network.py` for layer structure;
- `dqn_agent.py` for action selection and Double DQN learning;
- `environment.py` for policy-to-robot action mapping and action repeat;
- `robot_controller.py` for sensor ordering and wheel commands;
- `config.py` for all dimensions, limits, ratios, and hyperparameters.

If code and this document ever disagree, treat the executable code and the
saved run manifest as the experiment record, then update this document before
the next controlled training run.
