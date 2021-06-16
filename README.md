# Covid-19-Detector

The model is based on a two stage transfer learning method. First the COIVDX dataset is trained on Resnet50. The deep features are extracted and passed through a network with Squeeze Excitaion layers. This secondary network trained with resnet features predicts the final result.

## Architecture
![squeeze](https://user-images.githubusercontent.com/31564734/122194946-8c1b4f80-ceb3-11eb-9a2a-49a16666ceac.png)

The blocks are made up of 1x1 convolution, a squeeze excitation and adaptive average pooling layer.