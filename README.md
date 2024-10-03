

# RT-IoT2022 Dataset research

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


## Install
To use this reserach code:
```bash
git pull https://github.com/Bierchermuesli/sklearn-research
cd sklearn-research
pipenv shell 
pipenv install
```

## Learning Models
The dataset provides some inspiring examples, such as [this one](https://www.kaggle.com/code/navyeesh/kothoju-navyeesh-rt-iot2022) [Apache 2.0 ](https://www.apache.org/licenses/LICENSE-2.0). I decided to work with scikit-learn and adapt some examples and add some of my  enhancements:

 * add a OneHotEncoder for better handling with missing(?), categorical or numerical features
 * Included a binary label to differentiate between Normal and Attack traffic pattern
 * Created a dedicated encoder and preprocessor pipeline for reuse in prediction tasks
 * Implemented model saving functionality with joblib
 * Organized the code for better clarity and to accommodate additional models
 * Added some command-line arguments
 * a fancy spinner!

In the current sample, the following models are generated:
 * Linear Perceptron
 * RandomForest
 * A VotingClassifier combining Random Forest, Decision Tree, KNN, and MLP Classifier 

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

## Test Predictions
The prediction script utilizes the previously generated model, encoder, and preprocessor pipeline. I used the same data but arranged it in a variety of orders, excluding the Attack_type column.

Let's verify the data with a prepared CSV file that contains alternating Normal and Attack traffic patterns:

### Perceptron Model
```bash
python3 predict.py -vvv -t test_random.csv
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

#### ARP_poisioning debugging
Lets verify with only thid traffic pattern: 
```
csvgrep -c 85 -m ARP_poisioning -a dataset/RT_IOT2022.csv > test_arp_poisening.csv

python3 predict.py -t test_arp_poisening.csv
FYI - Attack_type labels removed
--------------------------------------------------------------------------------
#Some stats

Predicted_Attack_Type
ARP_poisioning    4742
Normal            3008

```
We see that 3008 flows are not regognized correctly
#### DOS_SYN_Hping debugging
```bash
csvgrep -c 85 -m DOS_SYN_Hping -a dataset/RT_IOT2022.csv > test_arp_dos_syn_hping.csv 
python3 predict.py -t test_arp_dos_syn_hping.csv
FYI - Attack_type labels removed
--------------------------------------------------------------------------------
# Prediction stats

Predicted_Attack_Type
DOS_SYN_Hping    94659
```
It looks good for this kind of patterns. I personal belive this is a dataset issue (see notes below) but can also be a model issue.


## RandomForest
Same bad rusults with the RandomForest model:

```bash
python3 predict.py -t test_arp_poisening.csv -m model_randomforest.pkl
FYI - Attack_type labels removed
--------------------------------------------------------------------------------
# Prediction stats

Predicted_Attack_Type
ARP_poisioning    4742
Normal            3008
```
## VotingClassifier
With the complex voting model, it looks much more precise, all attacks are corretly regognized: 
```bash
python3 predict.py -t test_random.csv -m model_ensemble.pkl -vv
FYI - Attack_type labels removed
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


# Personal Smmary

The 'single' generated model have troubles to detect certain attacks. It was possible to detect them with the fancy voting model - however I personal belive a the beter model is just able to handle the some bad training data. In my opinion this Dataset contains invalid data. This can be proofed by some simple assumptions:
   * `ARP_Poisening` contains `udp` and `tcp` traffic
   * `Metasploit_Brute_Force_SSH`: contains `udp/dns` and `tcp/http` traffic
   * `NMAP_UDP_SCAN`: contains `tcp/http` traffic
   * `DDOS_Slowloris`: contains `udp/dhcp`, `udp/dns`traffic
   * etc




2/3.10.2024 ~10h research


