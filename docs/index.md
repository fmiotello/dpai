---
layout: default
title: DPAI
---

## Abstract

In this manuscript we propose a novel method to perform audio inpainting, i.e. the restoration of audio signals presenting multiple missing parts. Audio inpainting can be interpreted in the context of inverse problems as the task of reconstructing an audio signal from its corrupted observation. For this reason, our solution is based on a deep prior approach, a recently proposed technique that proved to be effective in the solution of many inverse problems, among which image inpainting.
Deep prior allows to consider the structure of a neural network as an implicit prior and to adopt it as a regularizer. Differently from the classical deep learning paradigm, deep prior performs a single-element training and thus can be applied to corrupted audio signals independently from the available training data sets. In the context of audio inpainting, a network presenting relevant audio priors will possibly generate a restored version of an audio signal, only provided with its corrupted observation.
Our method exploits a time-frequency representation of audio signals and makes use of a multi-resolution convolutional autoencoder, that has been enhanced to perform the harmonic convolution operation. Results show that the proposed technique is able to provide a coherent and meaningful reconstruction of the corrupted audio. It is also able to outperform the methods considered for comparison, in its domain of application.

## Audio examples

### Class 1

<!--<div class="container">
   <div class="column-left">
     <h6>Source speaker</h6>
     <audio src="audio/class_3_source.wav" controls preload></audio>
   </div>
   <div class="column-center">
     <h6>Target speaker</h6>
     <audio src="audio/class_3_target.wav" controls preload></audio>
   </div>
   <div class="column-right">
     <h6>Output</h6>
     <audio src="audio/class_3_output.wav" controls preload></audio>
   </div>
</div>-->

### Class 2

<!--<div class="container">
   <div class="column-left">
     <h6>Source speaker</h6>
     <audio src="audio/class_4_source.wav" controls preload></audio>
   </div>
   <div class="column-center">
     <h6>Target speaker</h6>
     <audio src="audio/class_4_target.wav" controls preload></audio>
   </div>
   <div class="column-right">
     <h6>Output</h6>
     <audio src="audio/class_4_output.wav" controls preload></audio>
   </div>
</div>-->

### Class 3

<!--<div class="container">
   <div class="column-left">
     <h6>Source speaker</h6>
     <audio src="audio/class_5_source.wav" controls preload></audio>
   </div>
   <div class="column-center">
     <h6>Target speaker</h6>
     <audio src="audio/class_5_target.wav" controls preload></audio>
   </div>
   <div class="column-right">
     <h6>Output</h6>
     <audio src="audio/class_5_output.wav" controls preload></audio>
   </div>
</div>-->

### Class 4

<!--<div class="container">
   <div class="column-left">
     <h6>Source speaker</h6>
     <audio src="audio/class_1_source.wav" controls preload></audio>
   </div>
   <div class="column-center">
     <h6>Target speaker</h6>
     <audio src="audio/class_1_target.wav" controls preload></audio>
   </div>
   <div class="column-right">
     <h6>Output</h6>
     <audio src="audio/class_1_output.wav" controls preload></audio>
   </div>
</div>-->

### Class 5

<!--<div class="container">
   <div class="column-left">
     <h6>Source speaker</h6>
     <audio src="audio/class_2_source.wav" controls preload></audio>
   </div>
   <div class="column-center">
     <h6>Target speaker</h6>
     <audio src="audio/class_2_target.wav" controls preload></audio>
   </div>
   <div class="column-right">
     <h6>Output</h6>
     <audio src="audio/class_2_output.wav" controls preload></audio>
   </div>
</div>-->
