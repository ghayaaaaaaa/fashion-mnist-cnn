# Fashion-MNIST CNN

A small, command-line image-classification project that trains configurable convolutional neural networks on the Fashion-MNIST benchmark with TensorFlow/Keras.

## What it demonstrates

- Loads the public Fashion-MNIST dataset through Keras; no image files are bundled.
- Normalizes grayscale images and adds the channel dimension expected by a CNN.
- Offers compact, balanced, and deeper model variants through one CLI.
- Uses integer labels with sparse categorical cross-entropy and early stopping on validation loss.
- Evaluates the held-out test set and writes a JSON report with per-class metrics and a confusion matrix.
- Saves the trained Keras model and training history to a user-selected local output directory.

## Quick start

Use Python 3.10–3.13 and create an isolated environment.

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
fashion-mnist-train --variant balanced --epochs 15 --output-dir outputs/run-001
~~~

On macOS or Linux, activate the environment with:

~~~bash
source .venv/bin/activate
~~~

The first run may download Fashion-MNIST into the Keras cache. Training can take longer on CPU. Generated models, metrics, and history stay in the selected output directory, which is ignored by Git.

## CLI options

~~~text
fashion-mnist-train --help
~~~

- **--variant**: compact, balanced, or deep
- **--epochs**: maximum training epochs
- **--batch-size**: number of examples per batch
- **--patience**: epochs without validation-loss improvement before early stopping
- **--validation-split**: fraction of training data reserved for validation
- **--seed**: random seed used by Keras
- **--output-dir**: destination for the model and JSON reports

## Output files

- **fashion_mnist_cnn.keras**: trained model
- **evaluation.json**: test loss, test accuracy, per-class precision/recall/F1/support, and confusion matrix
- **training_history.json**: loss and accuracy values recorded during fitting

## Data and references

Fashion-MNIST contains 60,000 training and 10,000 test grayscale images across 10 classes. The dataset is held by Zalando SE and licensed under MIT. See [Third-party notices](THIRD_PARTY_NOTICES.md).

- [Keras Fashion-MNIST API](https://keras.io/api/datasets/fashion_mnist/)
- [TensorFlow installation guide](https://www.tensorflow.org/install/pip)
- [Zalando Research Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist)

## Provenance and results

This standalone portfolio refactor is based on the image-classification notebook supplied by Ghaya Fouad. The code organization and documentation were prepared with AI assistance. The source report and presentation were not copied into this repository, and no inherited accuracy or training result is claimed.

The training code was not executed while preparing this repository. Run it locally to generate results; no benchmark score is stated here.
