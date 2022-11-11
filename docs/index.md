---
layout: default
title: DPAI
---

## Abstract

In this manuscript we propose a novel method to perform audio inpainting, i.e. the restoration of audio signals presenting multiple missing parts. Audio inpainting can be interpreted in the context of inverse problems as the task of reconstructing an audio signal from its corrupted observation. For this reason, our solution is based on a deep prior approach, a recently proposed technique that proved to be effective in the solution of many inverse problems, among which image inpainting.
Deep prior allows to consider the structure of a neural network as an implicit prior and to adopt it as a regularizer. Differently from the classical deep learning paradigm, deep prior performs a single-element training and thus can be applied to corrupted audio signals independently from the available training data sets. In the context of audio inpainting, a network presenting relevant audio priors will possibly generate a restored version of an audio signal, only provided with its corrupted observation.
Our method exploits a time-frequency representation of audio signals and makes use of a multi-resolution convolutional autoencoder, that has been enhanced to perform the harmonic convolution operation. Results show that the proposed technique is able to provide a coherent and meaningful reconstruction of the corrupted audio. It is also able to outperform the methods considered for comparison, in its domain of application.

## Audio examples

<div class="container">
   <div class="column-1">
     <h6>Corrupted audio signal</h6>
     <audio src="audio/example0/audio_original_masked.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-2">
     <h6>DPAI reconstruction (ours)</h6>
     <audio src="audio/example0/dpai.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-3">
     <h6>CAW reconstruction</h6>
     <audio src="audio/example0/caw.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-4">
     <h6>SGA reconstruction</h6>
     <audio src="audio/example0/sga.wav" controls preload style="width: 190px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/example1/audio_original_masked.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/example1/dpai.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/example1/caw.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/example1/sga.wav" controls preload style="width: 190px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/example2/audio_original_masked.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/example2/dpai.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/example2/caw.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/example2/sga.wav" controls preload style="width: 190px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/example3/audio_original_masked.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/example3/dpai.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/example3/caw.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/example3/sga.wav" controls preload style="width: 190px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/example4/audio_original_masked.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/example4/dpai.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/example4/caw.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/example4/sga.wav" controls preload style="width: 190px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/example5/audio_original_masked.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/example5/dpai.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/example5/caw.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/example5/sga.wav" controls preload style="width: 190px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/example6/audio_original_masked.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/example6/dpai.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/example6/caw.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/example6/sga.wav" controls preload style="width: 190px;"></audio>
   </div>
</div>

<div class="container">
   <div class="column-1">
     <audio src="audio/example7/audio_original_masked.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-2">
     <audio src="audio/example7/dpai.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-3">
     <audio src="audio/example7/caw.wav" controls preload style="width: 190px;"></audio>
   </div>
   <div class="column-4">
     <audio src="audio/example7/sga.wav" controls preload style="width: 190px;"></audio>
   </div>
</div>
