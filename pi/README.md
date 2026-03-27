# 🛡️ 6G Intrusion Detection System (IDS)

A comprehensive machine learning-based intrusion detection system for 6G network slices, featuring an ensemble of XGBoost, Random Forest, and MLP models with an interactive Streamlit interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Datasets](#datasets)
- [Model Performance](#model-performance)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Attack Types](#attack-types)

---

## 🎯 Overview

This project implements an advanced intrusion detection system designed for 6G networks, supporting multiple network slices:
- **eMBB** (Enhanced Mobile Broadband)
- **mMTC** (Massive Machine-Type Communications)
- **URLLC** (Ultra-Reliable Low-Latency Communications)
- **TON_IoT** (Telemetry dataset of IoT)

The system uses an ensemble approach combining three powerful machine learning models to achieve high accuracy across different attack types.

---

## ✨ Features

### 🔍 Live Detection Tab
- Manual input of network flow features
- Real-time prediction: Benign / False Alarm / Attack Type
- Individual model probabilities (XGBoost, Random Forest, MLP)
- Interactive threat gauge visualization
- Confidence scoring with adjustable threshold

### 📊 Model Comparison Tab
- F1 Macro score comparison across all models and datasets
- Grouped bar charts for visual comparison
- Average ranking table
- Radar chart showing ensemble coverage
- Performance metrics for 5 models × 4 datasets

### 📁 Batch Analysis Tab
- CSV file upload for bulk classification
- Batch prediction with confidence scores
- Downloadable results
- Summary statistics and pie charts
- Attack type distribution visualization

---

## 📊 Datasets

### eMBB (Enhanced Mobile Broadband)
- **Features**: Dur, TotPkts, TotBytes, Rate, Load, Loss, pLoss, TcpRtt
- **Samples**: 5,808 (3,023 Benign, 2,785 Malicious)
- **Attack Types**: TCP SYN Flood, Bandwidth Saturation
- **Key Pattern**: Short duration + Few packets = Malicious

### mMTC (Massive Machine-Type Communications)
- **Features**: Dur, TotPkts, TotBytes, SrcBytes, pLoss, Load, Loss, Rate
- **Samples**: 4,615 (2,462 Benign, 2,153 Malicious)
- **Attack Types**: TCP SYN Scan, Slow-Rate Exhaustion, FIN Scan
- **Key Pattern**: Low rate (<7 pkt/s) + Few packets = Malicious

### URLLC (Ultra-Reliable Low-Latency)
- **Features**: Dur, TotPkts, TotBytes, SrcBytes, pLoss, TcpRtt, SynAck, Load
- **Samples**: 4,033 (1,572 Benign, 2,461 Malicious)
- **Attack Types**: UDP DDoS Flood, RST Injection, SLA Violation, Reconnaissance
- **Key Pattern**: High latency (TcpRtt > 0.05s) = Violation

### TON_IoT (Telemetry of IoT)
- **Features**: duration, src_bytes, dst_bytes, src_pkts, dst_pkts, conn_state, service, proto
- **Samples**: 211,043 (49,983 Benign, 160,986 Malicious)
- **Attack Types**: DDoS, DoS, Scanning, Password, Backdoor, Ransomware, Injection, XSS, MITM
- **Key Pattern**: 9 distinct attack types with unique signatures

---

## 🏆 Model Performance

### Ensemble Models (XGBoost + Random Forest + MLP)

| Model | TON_IoT | eMBB | mMTC | URLLC | Average |
|-------|---------|------|------|-------|---------|
| **XGBoost** | 0.9944 | 0.9469 | 0.9252 | 0.7089 | **0.8938** |
| **Random Forest** | 0.9931 | 0.9372 | 0.9234 | 0.6676 | **0.8803** |
| **MLP** | 0.9910 | 0.8993 | 0.9026 | 0.6558 | **0.8622** |
| Extra Trees | 0.9558 | 0.8888 | 0.9096 | 0.6424 | 0.8492 |
| Logistic Reg. | 0.8811 | 0.8478 | 0.8830 | 0.5360 | 0.7870 |

**Why This Ensemble?**
- **XGBoost**: Best overall performance (0.8938 avg F1)
- **Random Forest**: Excellent explainability and stability
- **MLP**: Neural network diversity for complex patterns

**Ensemble Strategy**: Average voting with 30% confidence threshold for attack classification

---

## 🚀 Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 6G-IDS
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train models and export**
   - Open `6G_IDS_updated.ipynb`
   - Run all cells up to and including cell 61
   - This will create:
     ```
     streamlit_app/models/ensemble_models.pkl
     streamlit_app/models/ensemble_preprocessors.pkl
     streamlit_app/models/ensemble_features.pkl
     ```

4. **Run the Streamlit app**
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```

5. **Access the app**
   - Open your browser to `http://localhost:8501`

---

## 💻 Usage

### Live Detection

1. Select a dataset (eMBB, mMTC, URLLC, or TON_IoT)
2. Enter network flow features manually
3. Click "🔍 Run Detection"
4. View prediction, confidence, and attack type

### Batch Analysis

1. Select a dataset
2. Upload a CSV file with matching features
3. Click "🚀 Run Batch Detection"
4. Download results with predictions and confidence scores

### Model Comparison

- View F1 scores across all models and datasets
- Compare ensemble models performance
- Analyze radar charts for coverage

---

## 🧪 Testing

Synthetic test examples are provided for each dataset:

### eMBB Testing
- **File**: `test_eMBB_SYNTHETIC.csv`
- **Documentation**: `EXEMPLES_eMBB_SYNTHETIC.md`
- **Examples**: 5 Benign + 8 Malicious
- **Expected Accuracy**: 100% (13/13)

### mMTC Testing
- **File**: `test_mMTC_SYNTHETIC.csv`
- **Documentation**: `EXEMPLES_mMTC_SYNTHETIC.md`
- **Examples**: 5 Benign + 8 Malicious
- **Expected Accuracy**: 92% (12/13)

### URLLC Testing
- **File**: `test_URLLC_SYNTHETIC.csv`
- **Documentation**: `EXEMPLES_URLLC_SYNTHETIC.md`
- **Examples**: 5 Benign + 8 Malicious
- **Expected Accuracy**: 85% (11/13)

### TON_IoT Testing
- **File**: `test_TON_IoT_SYNTHETIC.csv`
- **Documentation**: `EXEMPLES_TON_IoT_SYNTHETIC.md`
- **Examples**: 5 Benign + 9 Malicious (all 9 attack types)
- **Expected Accuracy**: 71% (10/14)

**Note**: TON_IoT has lower accuracy due to the model being trained on 76% malicious data, making it intentionally aggressive for security purposes.

---

## 📁 Project Structure

```
6G-IDS/
├── 6G_IDS_updated.ipynb          # Main training notebook
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── Data5G/                        # Datasets
│   ├── eMBB.csv
│   ├── mMTC.csv
│   ├── URLLC.csv
│   └── train_test_network.csv    # TON_IoT
├── streamlit_app/                 # Streamlit application
│   ├── app.py                     # Main app file
│   └── models/                    # Trained models (generated)
│       ├── ensemble_models.pkl
│       ├── ensemble_preprocessors.pkl
│       └── ensemble_features.pkl
├── test_eMBB_SYNTHETIC.csv        # eMBB test examples
├── test_mMTC_SYNTHETIC.csv        # mMTC test examples
├── test_URLLC_SYNTHETIC.csv       # URLLC test examples
├── test_TON_IoT_SYNTHETIC.csv     # TON_IoT test examples
├── EXEMPLES_eMBB_SYNTHETIC.md     # eMBB documentation
├── EXEMPLES_mMTC_SYNTHETIC.md     # mMTC documentation
├── EXEMPLES_URLLC_SYNTHETIC.md    # URLLC documentation
└── EXEMPLES_TON_IoT_SYNTHETIC.md  # TON_IoT documentation
```

---

## 🎯 Attack Types

### eMBB Attacks
- **TCP SYN Flood**: Short duration (<0.35s), high rate (>50 pkt/s)
- **Bandwidth Saturation**: Sustained flow, high load

### mMTC Attacks
- **TCP SYN Scan / Connection Flooding**: Few packets (<10), short duration (<1s)
- **Slow-Rate Resource Exhaustion**: Long duration (>3s), low load (<3000)
- **FIN Scan**: Other suspicious patterns

### URLLC Attacks
- **UDP DDoS Flood**: Instant packets (Dur=0, TotPkts=1)
- **RST Injection**: High latency (TcpRtt > 0.05s), few packets
- **SLA Violation / DoS**: Very high latency (TcpRtt > 0.1s)
- **Reconnaissance**: Long duration with moderate activity

### TON_IoT Attacks (9 Types)
1. **DDoS**: Massive packet flood (>1000 pkts in <2s)
2. **DoS**: High packet rate (>200 pkts in <5s)
3. **Scanning**: Rejected connections (REJ state)
4. **Password**: SSH/FTP brute force attempts
5. **Backdoor**: Long persistent connections (>60s)
6. **Ransomware**: Large data transfers (>100KB in >10s)
7. **Injection**: HTTP with large responses (dst > src * 2.5)
8. **XSS**: HTTP-based attacks
9. **MITM**: Ultra-short connections or non-TCP/UDP protocols

---

## 🔧 Configuration

### Adjustable Parameters

**Detection Threshold** (default: 30%)
- Predictions below this probability are flagged as "False Alarm"
- Adjust in the sidebar: 10% - 90%
- Lower threshold = more sensitive (more detections)
- Higher threshold = more specific (fewer false positives)

**Ensemble Weights**
- Currently using equal weights (1/3 each)
- Can be modified in `app.py` for custom weighting

---

## 📈 Performance Notes

### Model Characteristics

**eMBB**: Excellent performance (94.7% F1)
- Clear separation between benign and malicious
- Key features: Duration, TotPkts, Rate

**mMTC**: Very good performance (92.5% F1)
- Rate is the strongest indicator
- IoT-specific attack patterns

**URLLC**: Good performance (70.9% F1)
- Latency-sensitive detection
- Challenging due to minimal traffic patterns

**TON_IoT**: Outstanding performance (99.4% F1)
- Most complex with 9 attack types
- Aggressive detection (76% training data is malicious)
- High false positive rate is intentional for security

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📄 License

[Add your license information here]

---

## 👥 Authors

[Add author information here]

---

## 🙏 Acknowledgments

- Dataset sources: [Add dataset sources]
- 6G network slice specifications
- Machine learning frameworks: scikit-learn, XGBoost, TensorFlow

---

## 📞 Contact

[Add contact information here]
