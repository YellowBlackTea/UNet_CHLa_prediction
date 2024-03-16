# Prediction of Chlorophyll-a (CHL-a) using UNet and CNN

This project was conducted during a one-month internship at the LOCEAN lab. I extend my sincere gratitude to my mentor, Luther Ollier (`lollier`), whose guidance and instruction were invaluable throughout this endeavor.

# Motivation
Obtaining CHL-a data could be achieved through two methods:
- By boat, utilizing a probe to obtain precise values:
    - Pros: Accuracy
    - Cons: Time-intensive process
- Through satellite scanning of the entire globe:
    - Pros: Rapid data acquisition without the need for physical presence at each CHL-a site
    - Cons: Lack of precision and occasional inconsistency
 
The primary objective of this project was to predict or reconstruct CHL-a levels from satellite data, using physics data such as temperature and ocean speed, rather than relying solely on satellite-derived CHL-a data. Two neural network architectures were employed: a CNN and a UNet.

# Quick Results
UNet outperforms CNN in predicting accurate CHL-a levels, as it can capture finer details and certain physical phenomena associated with CHL-a. CNN, on the other hand, could only roughly approximate the shape of CHL-a, making it a simpler method for obtaining a general CHL-a estimate.

For a glimpse into the dataset used, please refer to the `Read_me.txt` file.

# Detailed information
For a more comprehensive understanding of the project, the report is available upon request. Please feel free to contact me for further details.
