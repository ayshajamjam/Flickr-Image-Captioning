# Flickr Image Captioning LSTM

## Introduction

This project was completed as part of my Columbia University NLP course.

**Task**

- (part 1) Create matrices of image representations using an off-the-shelf image encoder.
- (part 2) Read and preprocess the image captions.
- Write a generator function that returns one training instance (input/output sequence pair) at a time.
- Train an LSTM language generator on the caption data.
- Write a decoder function for the language generator.
- Add the image input to write an LSTM caption generator.
- Implement beam search for the image caption generator.

**Data:** flickr8k data

### Part 1: Image Encodings

**Inception V3 network:** off-the-shelf pre-trained image encoder. This model is a CNN used for object detection.

1.  Resize images: Convert Flikr images to be (299 x 299) pixels with RGB values between 0 and 1.0.
2.  Encode each image as a vector of size 2048 and store them in one big matrix (enc_train, enc_dev, enc_test)

### Part 2: Text (Caption) Data Preparation

Load image captions and generate training data for the generator model.

### Part 3: Basic Decoder Model

In this part, we train a model for text generation. We use a Bidirectional LSTM to encode the sequence from both directions and then predict the output





