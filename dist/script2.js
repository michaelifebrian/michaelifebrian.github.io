let objectDetector;
let tensor;

const SEQUENCE_AXIS = -2;

class PositionalEmbedding extends tf.layers.Layer {

    constructor(args: PositionalEmbeddingLayerArgs) {
        super(args);
        this.sequenceLength = args.sequenceLength;
        this.initializer = getInitializer(args.initializer || "glorotUniform");
    }

    override getConfig(): tf.serialization.ConfigDict {
        const config = {
            sequenceLength: this.sequenceLength,
            initializer: serializeInitializer(this.initializer)
        };

        const baseConfig = super.getConfig();

        return {...baseConfig, ...config};
    }

    override build(inputShape: tf.Shape | tf.Shape[]): void {
        const oneShape = getExactlyOneShape(inputShape);
        const featureSize = oneShape[oneShape.length - 1];

        this.positionalEmbeddings = this.addWeight(
            "embeddings",
            [this.sequenceLength, featureSize],
            this.dtype,
            this.initializer,
            undefined,
            true
        );

        super.build(inputShape);
    }

    override call(
        inputs: tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[]
    ): tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[] {
        return tf.tidy(() => {
            const tensor = getExactlyOneTensor(inputs);

            // TF.js does not support RaggedTensor atm.
            return this.trimAndBroadcastPositionEmbeddings(tensor.shape);
        });
    }

    trimAndBroadcastPositionEmbeddings(shape: tf.Shape): tf.Tensor {
        const inputLength = shape[shape.length - SEQUENCE_AXIS]!;
        const positionEmbeddings = tf.slice(
            this.positionalEmbeddings!.read(),
            [0, 0],
            [inputLength, -1]
        );

        return tf.broadcastTo(
            positionEmbeddings,
            [...shape].map(dim => dim == null ? -1 : dim)
        );
    }
}

async function detect() {
  let tensor = tf.browser.fromPixels(document.querySelector("img"));
  tensor = tf.image.resizeBilinear(tensor, [456, 456])
  tensor = tf.reshape(tensor, [1,456,456,3])
  document.querySelector(".result").textContent = "detecting caption...";
  if (!objectDetector) {
    objectDetector = await tf.loadLayersModel('tfjs/model.json');
  }
  document.querySelector(".result").textContent = "model loaded"
  let result = objectDetector.predict(tensor);
  console.log(result)
  document.querySelector(".result").textContent = result;
}


document.querySelector(".btn").addEventListener("click", () => {
  console.log("hi");
  detect();

});