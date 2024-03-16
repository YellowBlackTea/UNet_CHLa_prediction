# Prediction of Chlorophyll-a (CHL-a) using UNet and CNN

This project was realised during an one-month internship in LOCEAN lab. A big thanks to my tutor Luther O. (`lollier`) who helped me and teached me a lot.

# Motivation
There were two ways to get CHL-a data:
- by boat and using a probe to get its exact value:
    - pros: precise value
    - cons: time required to get data
- by satellite scanning the whole globe:
    - pros: fast and doesn't require someone to go on each CHL spot
    - cons: not precise and sometimes messy
 
The main task of this project was to predict or re-build CHL-a from satellite data, using data from Physics (Temperature, Speed of the Ocean, etc.) and not directly the satellite CHL-a data. Two main neural network architectures were used: a CNN and a UNet.

# Quick result
UNet is better to predict a correct CHL-a as it can learn its details or some Physics phenomena of the CHL-a. CNN was only able to learn roughly the shape of the CHL-a, thus this model is a simple method to get a rough idea of the CHL-a.

# More detailled information
The report is available only by contacting me. 
