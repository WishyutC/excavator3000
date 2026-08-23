import com.google.flatbuffers.FlatBufferBuilder;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.tensorflow.lite.schema.ActivationFunctionType;
import org.tensorflow.lite.schema.Buffer;
import org.tensorflow.lite.schema.BuiltinOperator;
import org.tensorflow.lite.schema.BuiltinOptions;
import org.tensorflow.lite.schema.FullyConnectedOptions;
import org.tensorflow.lite.schema.FullyConnectedOptionsWeightsFormat;
import org.tensorflow.lite.schema.Model;
import org.tensorflow.lite.schema.Operator;
import org.tensorflow.lite.schema.OperatorCode;
import org.tensorflow.lite.schema.SubGraph;
import org.tensorflow.lite.schema.Tensor;
import org.tensorflow.lite.schema.TensorType;

public final class TfliteDenseExporter {
    private static int tensor(
        FlatBufferBuilder builder, String name, int[] shape, int buffer
    ) {
        int shapeOffset = Tensor.createShapeVector(builder, shape);
        int nameOffset = builder.createString(name);
        return Tensor.createTensor(
            builder, shapeOffset, TensorType.FLOAT32, buffer, nameOffset,
            0, false, 0, 0
        );
    }

    private static int operator(
        FlatBufferBuilder builder, int input, int weights, int bias,
        int output, byte activation
    ) {
        int inputs = Operator.createInputsVector(
            builder, new int[] {input, weights, bias}
        );
        int outputs = Operator.createOutputsVector(builder, new int[] {output});
        int options = FullyConnectedOptions.createFullyConnectedOptions(
            builder, activation, FullyConnectedOptionsWeightsFormat.DEFAULT,
            false, false
        );
        return Operator.createOperator(
            builder, 0, inputs, outputs, BuiltinOptions.FullyConnectedOptions,
            options, 0, (byte) 0, 0, 0
        );
    }

    private static byte[] slice(byte[] source, int offset, int length) {
        byte[] result = new byte[length];
        System.arraycopy(source, offset, result, 0, length);
        return result;
    }

    private static byte[] buildModel(byte[] weights) {
        int[] floatCounts = {640, 64, 4096, 64, 192, 3};
        int expectedBytes = 0;
        for (int count : floatCounts) expectedBytes += count * 4;
        if (weights.length != expectedBytes) {
            throw new IllegalArgumentException(
                "Expected " + expectedBytes + " weight bytes, got " + weights.length
            );
        }

        FlatBufferBuilder builder = new FlatBufferBuilder(32768);
        int[] dataOffsets = new int[floatCounts.length];
        int position = 0;
        for (int index = 0; index < floatCounts.length; index++) {
            int byteCount = floatCounts[index] * 4;
            dataOffsets[index] = Buffer.createDataVector(
                builder, slice(weights, position, byteCount)
            );
            position += byteCount;
        }

        int[] buffers = new int[7];
        buffers[0] = Buffer.createBuffer(builder, 0);
        for (int index = 0; index < dataOffsets.length; index++) {
            buffers[index + 1] = Buffer.createBuffer(builder, dataOffsets[index]);
        }

        int[] tensors = {
            tensor(builder, "observation", new int[] {1, 10}, 0),
            tensor(builder, "dense1_weights", new int[] {64, 10}, 1),
            tensor(builder, "dense1_bias", new int[] {64}, 2),
            tensor(builder, "dense1_relu", new int[] {1, 64}, 0),
            tensor(builder, "dense2_weights", new int[] {64, 64}, 3),
            tensor(builder, "dense2_bias", new int[] {64}, 4),
            tensor(builder, "dense2_relu", new int[] {1, 64}, 0),
            tensor(builder, "output_weights", new int[] {3, 64}, 5),
            tensor(builder, "output_bias", new int[] {3}, 6),
            tensor(builder, "q_values", new int[] {1, 3}, 0)
        };
        int tensorsVector = SubGraph.createTensorsVector(builder, tensors);
        int inputsVector = SubGraph.createInputsVector(builder, new int[] {0});
        int outputsVector = SubGraph.createOutputsVector(builder, new int[] {9});

        int[] operators = {
            operator(builder, 0, 1, 2, 3, ActivationFunctionType.RELU),
            operator(builder, 3, 4, 5, 6, ActivationFunctionType.RELU),
            operator(builder, 6, 7, 8, 9, ActivationFunctionType.NONE)
        };
        int operatorsVector = SubGraph.createOperatorsVector(builder, operators);
        int graphName = builder.createString("excavator3000_race_v4");
        int graph = SubGraph.createSubGraph(
            builder, tensorsVector, inputsVector, outputsVector,
            operatorsVector, graphName
        );

        int opcode = OperatorCode.createOperatorCode(
            builder, BuiltinOperator.FULLY_CONNECTED, 0, 1
        );
        int opcodesVector = Model.createOperatorCodesVector(
            builder, new int[] {opcode}
        );
        int subgraphsVector = Model.createSubgraphsVector(builder, new int[] {graph});
        int buffersVector = Model.createBuffersVector(builder, buffers);
        int description = builder.createString(
            "Excavator3000 curriculum-v4 race DQN; float32; outputs are Q-values"
        );
        int model = Model.createModel(
            builder, 3, opcodesVector, subgraphsVector, description,
            buffersVector, 0, 0
        );
        Model.finishModelBuffer(builder, model);
        return builder.sizedByteArray();
    }

    private static float[] floats(Buffer buffer) {
        ByteBuffer data = buffer.dataAsByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[data.remaining() / 4];
        for (int index = 0; index < values.length; index++) {
            values[index] = data.getFloat();
        }
        return values;
    }

    private static float[] dense(
        float[] input, float[] weights, float[] bias, int outputs, boolean relu
    ) {
        int inputs = input.length;
        float[] result = new float[outputs];
        for (int row = 0; row < outputs; row++) {
            float value = bias[row];
            int offset = row * inputs;
            for (int column = 0; column < inputs; column++) {
                value += input[column] * weights[offset + column];
            }
            result[row] = relu && value < 0.0f ? 0.0f : value;
        }
        return result;
    }

    private static void validateStructure(Model model) {
        if (model.version() != 3 || model.subgraphsLength() != 1) {
            throw new IllegalStateException("Unexpected TFLite model structure.");
        }
        if (model.operatorCodesLength() != 1
            || model.operatorCodes(0).builtinCode() != BuiltinOperator.FULLY_CONNECTED) {
            throw new IllegalStateException("Expected FULLY_CONNECTED operator code.");
        }
        SubGraph graph = model.subgraphs(0);
        if (graph.inputsLength() != 1 || graph.outputsLength() != 1
            || graph.operatorsLength() != 3 || graph.tensorsLength() != 10) {
            throw new IllegalStateException("Unexpected graph input/output dimensions.");
        }
        if (!"observation".equals(graph.tensors(graph.inputs(0)).name())
            || !"q_values".equals(graph.tensors(graph.outputs(0)).name())) {
            throw new IllegalStateException("Unexpected input or output tensor name.");
        }
    }

    private static String verify(Model model, Path vectorsPath) throws IOException {
        float[] w0 = floats(model.buffers(1));
        float[] b0 = floats(model.buffers(2));
        float[] w1 = floats(model.buffers(3));
        float[] b1 = floats(model.buffers(4));
        float[] w2 = floats(model.buffers(5));
        float[] b2 = floats(model.buffers(6));
        List<String> lines = Files.readAllLines(vectorsPath);
        double maximumError = 0.0;
        double errorSum = 0.0;
        int valueCount = 0;
        int actionMatches = 0;

        for (String line : lines) {
            String[] fields = line.split(",");
            if (fields.length != 13) throw new IllegalArgumentException("Bad vector row.");
            float[] input = new float[10];
            float[] expected = new float[3];
            for (int i = 0; i < 10; i++) input[i] = Float.parseFloat(fields[i]);
            for (int i = 0; i < 3; i++) expected[i] = Float.parseFloat(fields[i + 10]);
            float[] h0 = dense(input, w0, b0, 64, true);
            float[] h1 = dense(h0, w1, b1, 64, true);
            float[] actual = dense(h1, w2, b2, 3, false);
            int expectedAction = 0;
            int actualAction = 0;
            for (int i = 0; i < 3; i++) {
                double error = Math.abs(actual[i] - expected[i]);
                maximumError = Math.max(maximumError, error);
                errorSum += error;
                valueCount++;
                if (expected[i] > expected[expectedAction]) expectedAction = i;
                if (actual[i] > actual[actualAction]) actualAction = i;
            }
            if (expectedAction == actualAction) actionMatches++;
        }
        return String.format(
            java.util.Locale.ROOT,
            "VERIFY cases=%d max_abs_error=%.9g mean_abs_error=%.9g action_agreement=%.9g",
            lines.size(), maximumError, errorSum / valueCount,
            (double) actionMatches / lines.size()
        );
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            throw new IllegalArgumentException(
                "Usage: TfliteDenseExporter weights.bin output.tflite vectors.csv"
            );
        }
        byte[] tflite = buildModel(Files.readAllBytes(Path.of(args[0])));
        Files.write(Path.of(args[1]), tflite);
        ByteBuffer data = ByteBuffer.wrap(tflite).order(ByteOrder.LITTLE_ENDIAN);
        if (!Model.ModelBufferHasIdentifier(data)) {
            throw new IllegalStateException("Missing TFL3 FlatBuffer identifier.");
        }
        Model model = Model.getRootAsModel(data);
        validateStructure(model);
        System.out.println(verify(model, Path.of(args[2])));
    }
}
