# Money-Bill-Classification
Final Project - Electrical Engineering - Tel Aviv University
<div class='demo'>


[![Everything Is AWESOME](images/youtube.png)](https://www.youtube.com/watch?v=NaQBIpFaf9E)
</div>
<div class='Introduction'>
  
  <h1>
    Introduction
  </h1>
 
  
  <p>
		In our final project we focused on the fields of image processing, machine learning and deep learning.<br>
    We tried to help other with the knowledge we acquired during our studies.<br>
    C-BILL is a wearable device that enables visually impaired to be aware of the type of bill he/she is holding.
  </p>
  <img src = "images/model.JPG">
  <img src = "images/on_me.JPG" width="300">
</div>


<div class='Implementation'>
    <h1>
      Implementation
    </h1>
  <h3>
    C-BILL block diagram: 
  </h3>
      <img src = "images/block_diagram.JPG">
  </div>

  
<div class='data'>
  <h1>
    Data
  </h1>
  <p>
    The dataset was created thanks to mass recruitment and augmentation.<br>
    Train: ~1200 images.<br>
    Test : 258 images (different people from train)
  </p>
  <img src = "images/dataset.JPG">
</div>

<div class='results'>
  <h1>
    Results
  </h1>
  <h3>
    Transfer learning based on pre-trained deep feature extractor + SVM:
  </h3>
  <img src = "images/svm_dnn.JPG">
  <pre>
  *Features vector is the network last output before the first Fully Connected layer.
  Those features contain all the information learned by the neural network with respect to the input image.
</pre>
 
  <h3>
    Final Results:
  </h3>
  <img src = "images/confusion_matrix.JPG">
  
  <table style="width:30%">
  <tr>
    <th>Test Set</th>
    <th >Real Time</th>
  </tr>
  <tr>
    <td>95.34%</td>
    <td>90%</td>
   
  </tr>
</table>
</div>
