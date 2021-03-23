Instructions:
 - Download data from https://drive.google.com/file/d/1bGZ2yCjZ1ZsCSA5TJSJpACDhI1U9ChsR/view?usp=sharing and extract to this directory, merging 'results' folders.
 - The 'scripts' folder contains the scripts used to generate plots in the paper. It should be possible to run them after extracting data.
   - mnist_3layer_errors.py generates figure 2 and saves it as out/mnist_3layer_error.pdf.
   - mnist_3layer_rates.py generates figure 3 and saves it as out/mnist_3layer_rates.pdf.
   - mnist_5layer_sweep.py generates figure 4 and saves it as out/mnist_5layer_sweep.pdf.
   - mnist_5layer_dw.py generates the images used to create figure 5 and saves it as the files out/mnist_5layer_dW__mlff.png, out/mnist_5layer_dW__swni.png, and out/mnist_5layer_dW__leg.png.
 - Raw data is contained in the results folder. Refer to github.com/jgammell/equilibrium_propagation to see how it is generated, what it contains, and how to parse it. The above scripts also serve as examples of how to parse it.
 - The contents of table 1 were created using the contents of the log.txt files in the results/<trial> folder associated with each trial. These files list the training and test error rates for each epoch of training, and the log-spread after training.