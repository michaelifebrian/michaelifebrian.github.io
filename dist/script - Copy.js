let objectDetector;
let tensor;

async function detect() {
  //tf.setBackend("cpu")
  let tensor = tf.browser.fromPixels(document.querySelector("img"));
  tensor = tf.image.resizeBilinear(tensor, [456, 456])
  tensor = tf.reshape(tensor, [1,456,456,3])
  document.querySelector(".result").textContent = "detecting caption...";
  if (!objectDetector) {
    objectDetector = await tf.loadGraphModel('encoder_tfjsmodel/model.json');
  }
  document.querySelector(".result").textContent = "model loaded"
  console.log(tensor)
  //let result = objectDetector.predict(tensor)
  let result = objectDetector.predict(tensor);
  console.log(result)
  //document.querySelector(".result").textContent = result;
}


document.querySelector(".btn").addEventListener("click", () => {
  console.log("hi");
  detect();

});