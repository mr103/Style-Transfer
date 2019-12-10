File organization:

 Neural Style Transfer – Folder contains code for neural style transfer proposed by Gatys et al.
– demo.ipynb – Run a demo of the code
– train.ipynb – Run an example of the code
– utilis.py – Module for model, methods and dataset
– visu.py – Module for visualizing images and results

 Real-time Style Transfer – Folder contains code, resources and results for real-time style transfer
– Code - Folder contains code for real-time style transfer
 demo_rtst.ipynb – Run a demo of the code
 train_rtst.py – Run the program for loading dataset, training saving a model
 architecture.py – Module for classes
 utils_rtst.py – Module for visualizing results
 nntools_RTST.py – Module for implementing checking point for training
 output_generator.ipynb – Saves content, style and transformed images
– Models - Folder contains trained style models
 transform_monet – Claude Monet’s Oak fontainebleau style
 transform_starry – Vincent Van Gogh’s Starry night style
 transform_irises – Vincent Van Gogh’s Irises style
 transform_cezanne – Paul Cezanne’s forest style
– Images - Folder contains style, content, transformed images (naming convention: style images have prefix ‘s_’, content
images have prefix ‘c_’, transformed images: style name + content image name)