# Data
This data is published in the paper of ***Neural User Factor Adaptation for Text Classification: Learning to Generalize across Author Demographics.***

***WARNING: The data has been anonymized and should only be used for academic usage. The people who use this publicized data should be responsible for their illegal violations.***

## Data Details
 1. Four data sources: 
    * Twitter flu vaccine data;
    * Amazon Music Product Review;
    * Yelp Hotel and Restaurant Reviews.
2. Two tsv files are created for each data source (null values are set as **x**):
    * user_table.tsv
        - **uid**: user unique id;
        - **country**: binary values of whether the user is in the US (1) or not (0);
        - **region**: four different regions in the US, south (0), west (1), midwest (2) and northeast (3); Refer to the [US Census Bureau regional definition](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf).
        - **gender**: binary values, female (1) or male (0)
        - **age**: binary values of whether the use is under 31 (<= 30, 1) or not (> 30, 0).
    * `name`.tsv (replace with any data source name): 
        - **did**: unique id of the text document;
        - **content**: text content;
        - **time**: the created date of the text content, format: yyyy-mm-dd;
        - **uid**: unique user id;
        - **country**: same to the user_table.tsv;
        - **region**: same to the user_table.tsv;
        - **gender**: same to the user_table.tsv;
        - **age**: same to the user_table.tsv;
        - **label**: binary document labels. For the Twitter data the binary labels indicate if the user taken/plan to take flu vaccination (1) vs. not (0). For the other reviews data, the binary labels means positive (>3, 1) or negative (<=3, 0) attitudes.

***Please cite our paper in any published work that uses any of these resources.***
~~~
Xiaolei Huang and Michael J. Paul. Neural user factor adaptation for text classification: Learning to generalize across author demographics. Conference on Lexical and Computational Semantics (*SEM), Minneapolis, Minnesota. June 2019.
~~~

## Contact
Any further questions please contact with Xiaolei Huang: [xiaolei.huang@colorado.edu](mailto:xiaolei.huang@colorado.edu)
