
# ML / Dataset research

This repo has some sample code to test and compare datasets.


I decided to work with scikit-learn and adopted some model samples  [from here](https://www.kaggle.com/code/navyeesh/kothoju-navyeesh-rt-iot2022) ([Apache 2.0 ](https://www.apache.org/licenses/LICENSE-2.0)) I added some of my  enhancements:

  * add a OneHotEncoder for better handling with missing(?), categorical or numerical features
 * Included a binary label to differentiate between Normal and Attack traffic pattern
 * Created a dedicated encoder and preprocessor pipeline for reuse in prediction tasks
 * Implemented model saving functionality with joblib
 * Organized the code for better clarity and to accommodate additional models
 * Added some command-line arguments
 * data normalization
 * per dataset options
 * a fancy spinner!
 
 currently basic models are in use. 

In the current sample, the following models are generated:
 * Linear Perceptron
 * RandomForest
 * A VotingClassifier combining Random Forest, Decision Tree, KNN, and MLP Classifier

## Install
To use this reserach code:
```bash
git pull https://github.com/Bierchermuesli/sklearn-research
cd sklearn-research
pipenv shell 
pipenv install
```


## RT-IoT2022 Dataset research

This is a ML training Proof of concept with RT-IoT2022 Dataset. This dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and can be found on [Kaggle](https://www.kaggle.com/datasets/supplejade/rt-iot2022real-time-internet-of-things) or its origin from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/942/rt-iot2022). 

The dataset should include:
* 9 Attack scenarios:
    * DOS_SYN_Hping
    * ARP_poisioning
    * NMAP_UDP_SCAN
    * NMAP_XMAS_TREE_SCAN
    * NMAP_OS_DETECTION
    * NMAP_TCP_scan
    * DDOS_Slowloris
    * Metasploit_Brute_Force_SSH
    * NMAP_FIN_SCAN
* and 3 normal pattern:
    * MQTT
    * Thing_speak
    * Wipro_bulb_Dataset
    * (according to authors also Amazon-Alexa but it is missing)


### Trainig Models
 

```bash
python3 learn.py 
Prepping the RandomForest
Accuracy: 0.9723
Saving model to model_perceptron.pkl
✔ Create Perceptron Model
Prepping the RandomForest
Accuracy: 0.9991
Saving model to model_randomforest.pkl
✔ Create RandomForest Model
Prepping the VotingClassifier
Accuracy: 0.9987
Saving model to model_ensemble.pkl
✔ Create Ensemble Model

```

### Test Predictions
The prediction script utilizes the previously generated model, encoder, and preprocessor pipeline. I used the same data but arranged it in a variety of orders, excluding the Attack_type column.

Let's verify the data with a prepared CSV file that contains alternating Normal and Attack traffic patterns:

#### Perceptron Model
```bash
python3 predict.py -vvv -d test_random.csv
--------------------------------------------------------------------------------
# Expected Labels:

0                   MQTT_Publish
1              NMAP_OS_DETECTION
2                   MQTT_Publish
3                  NMAP_UDP_SCAN
4                   MQTT_Publish
5     Metasploit_Brute_Force_SSH
6                   MQTT_Publish
7                  DOS_SYN_Hping
8                   MQTT_Publish
9                 ARP_poisioning
10                  MQTT_Publish
11           NMAP_XMAS_TREE_SCAN
12                  MQTT_Publish
13                 NMAP_TCP_scan
14                  MQTT_Publish
15                DDOS_Slowloris
16                  MQTT_Publish
17                 NMAP_FIN_SCAN
18                  MQTT_Publish
19                  MQTT_Publish
20                  MQTT_Publish
21                  MQTT_Publish
22                  MQTT_Publish
23                  MQTT_Publish
24                  MQTT_Publish
25                  MQTT_Publish
26                  MQTT_Publish
27                  MQTT_Publish
28                    Wipro_bulb
29                   Thing_Speak
30                ARP_poisioning
31                ARP_poisioning
32                ARP_poisioning

--------------------------------------------------------------------------------
# Prediction stats

   Predicted_Binary_Label       Predicted_Attack_Type
0                  Normal                      
1                  Attack           NMAP_OS_DETECTION
2                  Normal                      
3                  Attack               NMAP_UDP_SCAN
4                  Normal                      
5                  Attack  Metasploit_Brute_Force_SSH
6                  Normal                      
7                  Attack               DOS_SYN_Hping
8                  Normal                      
9                  Normal                      
10                 Normal                      
11                 Attack         NMAP_XMAS_TREE_SCAN
12                 Normal                      
13                 Attack               NMAP_TCP_scan
14                 Normal                      
15                 Attack              DDOS_Slowloris
16                 Normal                      
17                 Attack               NMAP_FIN_SCAN
18                 Normal                      
19                 Normal                      
20                 Normal                      
21                 Normal                      
22                 Normal                      
23                 Normal                      
24                 Normal                      
25                 Normal                      
26                 Normal                      
27                 Normal                      
28                 Normal                      
29                 Normal                      
30                 Normal                      
31                 Normal                      
32                 Attack              ARP_poisioning
Predicted_Attack_Type
ARP_poisioning                 1
DDOS_Slowloris                 1
DOS_SYN_Hping                  1
Metasploit_Brute_Force_SSH     1
NMAP_FIN_SCAN                  1
NMAP_OS_DETECTION              1
NMAP_TCP_scan                  1
NMAP_UDP_SCAN                  1
NMAP_XMAS_TREE_SCAN            1
Normal                        24

Results are also saved to result.csv

```
This works good so far. As we can see. ARP_poisening is not correctly regognized. 

##### ARP_poisioning debugging
Lets verify with only this traffic pattern: 
```
csvgrep -c 85 -m ARP_poisioning -a trainset/RT_IOT2022.csv > test_arp_poisening.csv

python3 predict.py -t test_arp_poisening.csv
Labels removed!
--------------------------------------------------------------------------------
#Some stats

Predicted_Attack_Type
ARP_poisioning    4742
Normal            3008

```
We see that 3008 flows are not regognized correctly
##### DOS_SYN_Hping debugging
```bash
csvgrep -c 85 -m DOS_SYN_Hping -a trainset/RT_IOT2022.csv > test_arp_dos_syn_hping.csv 
python3 predict.py -t test_arp_dos_syn_hping.csv
Labels removed!
--------------------------------------------------------------------------------
# Prediction stats

Predicted_Attack_Type
DOS_SYN_Hping    94659
```
It looks good for this kind of patterns. I personal belive this is a dataset issue (see notes below) but can also be a model issue.


### RandomForest
Same bad rusults with the RandomForest model:

```bash
python3 predict.py -t test_arp_poisening.csv -m model_randomforest.pkl
Labels removed!
--------------------------------------------------------------------------------
# Prediction stats

Predicted_Attack_Type
ARP_poisioning    4742
Normal            3008
```
### VotingClassifier
With the complex voting model, it looks much more precise, all attacks are corretly regognized: 
```bash
python3 predict.py -t test_random.csv -m model_ensemble.pkl -vv
Labels removed!
--------------------------------------------------------------------------------
# Prediction stats

   Predicted_Binary_Label       Predicted_Attack_Type
0                  Normal                      
1                  Attack           NMAP_OS_DETECTION
2                  Normal                      
3                  Attack               NMAP_UDP_SCAN
4                  Normal                      
5                  Attack  Metasploit_Brute_Force_SSH
6                  Normal                      
7                  Attack               DOS_SYN_Hping
8                  Normal                      
9                  Attack              ARP_poisioning
10                 Normal                      
11                 Attack         NMAP_XMAS_TREE_SCAN
12                 Normal                      
13                 Attack               NMAP_TCP_scan
14                 Normal                      
15                 Attack              DDOS_Slowloris
16                 Normal                      
17                 Attack               NMAP_FIN_SCAN
18                 Normal                      
19                 Normal                      
20                 Normal                      
21                 Normal                      
22                 Normal                      
23                 Normal                      
24                 Normal                      
25                 Normal                      
26                 Normal                      
27                 Normal                      
28                 Normal                      
29                 Normal                      
30                 Attack              ARP_poisioning
31                 Attack              ARP_poisioning
32                 Attack              ARP_poisioning
Predicted_Attack_Type
ARP_poisioning                 4
DDOS_Slowloris                 1
DOS_SYN_Hping                  1
Metasploit_Brute_Force_SSH     1
NMAP_FIN_SCAN                  1
NMAP_OS_DETECTION              1
NMAP_TCP_scan                  1
NMAP_UDP_SCAN                  1
NMAP_XMAS_TREE_SCAN            1
Normal                        21

Results are also saved to result.csv
```


### Personal Smmary

The 'single' generated model have troubles to detect certain attacks. It was possible to detect them with the fancy voting model - however I personal belive a the beter model is just able to handle the some bad training data. In my opinion this Dataset contains invalid data. This can be proofed by some simple assumptions:
   * `ARP_Poisening` contains `udp` and `tcp` traffic
   * `Metasploit_Brute_Force_SSH`: contains `udp/dns` and `tcp/http` traffic
   * `NMAP_UDP_SCAN`: contains `tcp/http` traffic
   * `DDOS_Slowloris`: contains `udp/dhcp`, `udp/dns`traffic
   * etc


## ToN IoT

The [ToN_IoT](https://research.unsw.edu.au/projects/toniot-datasets) dataset is available for free use for academic research purposes.

In a theoretical review, this dataset reveals some inaccuracies and questionable testbed methods. Nevertheless, it can still offer valuable data. In practice, this dataset is not entirely clean and contains too many unnecessary features. The common denominator among these features is relatively low, which raises questions about their value. In this scratch pad section, I would like to focus on the limitations of these features and measure their relevance.
To address these issues, I introduced a feature filter and implemented data cleaning and normalization procedures. This revision corrects minor errors, improves sentence structure, and enhances overall readability.

### Train with all features

Trainig with all (44) features:

```bash
python learn.py -d trainset/Ton_IoT_train_test_network.csv -vv --all-features

Normal/Evil Ratio:
   LABEL_BOOL   Count  Percentage
          1  161043   76.308146
          0   50000   23.691854

Attack Type Ratio:
        LABEL  Count  Percentage
       normal  50000   23.691854
     backdoor  20000    9.476742
         ddos  20000    9.476742
          dos  20000    9.476742
    injection  20000    9.476742
     password  20000    9.476742
     scanning  20000    9.476742
   ransomware  20000    9.476742
          xss  20000    9.476742
         mitm   1043    0.494212


Rows: 211043
Features: 44
 - src_ip
 - src_port
 - dst_ip
 - dst_port
 - proto
 - service
 - duration
 - src_bytes
 - dst_bytes
 - conn_state
 - missed_bytes
 - src_pkts
 - src_ip_bytes
 - dst_pkts
 - dst_ip_bytes
 - dns_query
 - dns_qclass
 - dns_qtype
 - dns_rcode
 - dns_AA
 - dns_RD
 - dns_RA
 - dns_rejected
 - ssl_version
 - ssl_cipher
 - ssl_resumed
 - ssl_established
 - ssl_subject
 - ssl_issuer
 - http_trans_depth
 - http_method
 - http_uri
 - http_version
 - http_request_body_len
 - http_response_body_len
 - http_status_code
 - http_user_agent
 - http_orig_mime_types
 - http_resp_mime_types
 - weird_name
 - weird_addl
 - weird_notice
 - LABEL_BOOL
 - LABEL

Labels: 10
- backdoor
- ddos
- dos
- injection
- mitm
- normal
- password
- ransomware
- scanning
- xss
Create models:
✔ perceptron crated. Accuracy: 0.6592 - 3.1s 
✔ randomforest created. Accuracy: 0.9879 - 13.5s
✔ ensemble created. Accuracy: 0.951551 - 716.4s
```


### Train with limited features

With the filter enabled (default), all `http_*`,`ssl_*`,`weird_*`,`dns_*` features are removed: 

```bash
python learn.py -d trainset/Ton_IoT_train_test_network.csv -vv
Rows: 211043
Features: 17
src_ip
 - src_port
 - dst_ip
 - dst_port
 - proto
 - service
 - duration
 - src_bytes
 - dst_bytes
 - conn_state
 - missed_bytes
 - src_pkts
 - src_ip_bytes
 - dst_pkts
 - dst_ip_bytes
 - LABEL_BOOL
 - LABEL

Labels: 10
- backdoor
- ddos
- dos
- injection
- mitm
- normal
- password
- ransomware
- scanning
- xss
Create models:
✔ perceptron created. Accuracy: 0.6234 - 2.3s
✔ randomforest created. Accuracy: 0.9874 - 14.2s
✔ ensemble created Accuracy: 0.961193 - 675.2s
```

#### Training Summary:
 - a reduced features set (44 vs 17 features) does not change the outcome of the training data (or at least it doesn't make it worse)
 - the accuracy of each model is equal to 44 vs 17 feature:
   - good rate on RandomForest and VotingClassifier esembled model
   - poor on linear perceptron (expected)
   - However: these are sill basic model without any tweaks


### Predicting full datasets: 

The dataset is splittet into 1Mio records per file. The file itself is very un-shuffled, each file is very inbalanced with a big Normal/Evil Ratio, usually each file contains one attack type. 

<details>
<summary>Network_dataset_1.csv: Accuracy: 79%</summary>

```python
python3 predict.py -m model_ensemble.pkl -d Network_dataset_1.csv -vv

Labled Normal/Evil Ratio:
   LABEL_BOOL   Count  Percentage
      True     791321     79.1321
      False    208679     20.8679

 Labeled Type Ratio:
      LABEL   Count  Percentage
   scanning  791321     79.1321
     normal  208679     20.8679

✔ Labels removed!

Rows: 1000000
Labels Encoded:
backdoor
- ddos
- dos
- injection
- mitm
- normal
- password
- rdetailansomware
- scanning
- xss

Prediction Summary and Ratio:
      PREDICT   Count  Percentage
     scanning  642789     64.2789
       normal  245669     24.5669
   ransomware   66549      6.6549
          dos   12906      1.2906
     backdoor   12021      1.2021
         ddos   11767      1.1767
          xss    3600      0.3600
         mitm    2510      0.2510
     password    1367      0.1367
    injection     822      0.0822


Results saved to result.csv

Accuracy of predictions: 0.7970

Duration: 357.4s
```
</details>

<details>
<summary>Network_dataset_3.csv: Accuracy: 87%</summary>

```python
python3 predict.py -m model_ensemble.pkl -d Network_dataset_3.csv -vv
Fix dataypes and normalize values

Labled Normal/Evil Ratio:
 LABEL_BOOL  Count  Percentage
       True 997180      99.718
      False   2820       0.282

Labeled Type Ratio:
   LABEL  Count  Percentage
scanning 997180      99.718
  normal   2820       0.282

✔ Labels removed!

Rows: 1000000 loaded
Labels Encoded: 10
 - backdoor
 - ddos
 - dos
 - injection
 - mitm
 - normal
 - password
 - ransomware
 - scanning
 - xss
duration: 363.3s

Prediction Summary and Ratio:
   PREDICT  Count  Percentage
  scanning 871475     87.1475
      ddos  54264      5.4264
       xss  42550      4.2550
ransomware   8058      0.8058
  password   5299      0.5299
    normal   5126      0.5126
       dos   4620      0.4620
  backdoor   4313      0.4313
 injection   3538      0.3538
      mitm    757      0.0757

Results saved to result.csv
Accuracy of predictions: 0.8743
```
</details>


### Predicting Summary:
 - Data nomalization and cleaning is nessessary due bad data and for performance improovment. 
 - 70-90% accuraty on full datasets - not very good
 - The datasets(-files) are unbalanced 
 - The models are too basic, tweaking is nessessary
 - quick tries without success:
   - RandomForestClassifier/DecisionTreeClassifier: `class_weight='balanced'`
   - VotingClassifier: `voting="soft"`
   - MLPClassifier: `max_iter=500` due `ConvergenceWarning: Stochastic Optimizer: Maximum Iterations (200) reached and the optimization hasn't converged yet.` warnings
      - Learning takes 30min...
 