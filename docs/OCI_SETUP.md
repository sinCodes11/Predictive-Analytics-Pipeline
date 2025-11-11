# Oracle Cloud Infrastructure (OCI) Setup Guide

Complete guide to deploying the Customer Churn Prediction pipeline on OCI's **Always Free Tier** (no credit card charges).

## What's Available in OCI Free Tier

### Always Free Resources (No Time Limit)

✅ **Compute**
- 2x AMD-based Compute VMs (1/8 OCPU, 1 GB memory each)
- OR 4x Arm-based Ampere A1 cores, 24 GB memory (flexible allocation)

✅ **Storage**
- 200 GB total Block Volume storage
- 10 GB Object Storage
- 10 GB Archive Storage

✅ **Database**
- 2x Oracle Autonomous Databases (20 GB each)

✅ **Networking**
- 1 public IP address
- Outbound data transfer: 10 TB/month

✅ **Additional Services**
- Load Balancer (1 instance, 10 Mbps)
- Monitoring and notifications
- VCN (Virtual Cloud Network)

## Deployment Options

### Option 1: Serverless with OCI Functions (Recommended)

**Best for**: Production API with automatic scaling

**Costs**: $0/month (up to 2M invocations)

#### Setup Steps

1. **Install OCI CLI**
```bash
# macOS (you already have it installed!)
# Check version
oci --version

# Configure (one-time setup)
oci setup config
```

2. **Create Function Application**
```bash
# Login to OCI Console: cloud.oracle.com
# Navigate to: Developer Services > Functions

# Create Application
# - Name: churn-prediction-api
# - VCN: Select your VCN
# - Subnet: Select public subnet

# Or via CLI:
oci fn application create \
  --compartment-id <your-compartment-id> \
  --display-name churn-prediction-api \
  --subnet-ids '["<subnet-ocid>"]'
```

3. **Deploy Function**
```bash
cd /path/to/predictive-analytics-pipeline

# Create func.yaml
cat > func.yaml <<EOF
schema_version: 20180708
name: churn-prediction
version: 0.0.1
runtime: python
entrypoint: /python/bin/fdk /function/func.py handler
memory: 512
EOF

# Create func.py (serverless handler)
cat > func.py <<'EOF'
import io
import json
import joblib
import pandas as pd
from fdk import response

# Load model (cached after first invocation)
model = None
feature_engineer = None

def load_artifacts():
    global model, feature_engineer
    if model is None:
        model = joblib.load('models/churn_model.pkl')
        # Load feature engineer
        from feature_engineering import ChurnFeatureEngineer
        feature_engineer = ChurnFeatureEngineer()
        feature_engineer.load('models/feature_engineer.pkl')

def handler(ctx, data: io.BytesIO = None):
    load_artifacts()

    try:
        body = json.loads(data.getvalue())
        df = pd.DataFrame([body])
        X, _ = feature_engineer.prepare_features(df, fit=False)

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]

        return response.Response(
            ctx,
            response_data=json.dumps({
                "churn_probability": float(probability),
                "risk_score": int(probability * 100),
                "will_churn": bool(prediction)
            }),
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        return response.Response(
            ctx,
            response_data=json.dumps({"error": str(e)}),
            status_code=500
        )
EOF

# Deploy
fn deploy --app churn-prediction-api
```

4. **Invoke Function**
```bash
# Test locally
fn invoke churn-prediction-api churn-prediction < data/sample_customer.json

# Get function endpoint
oci fn function list --application-id <app-id>

# Invoke via HTTP
curl -X POST <function-endpoint> \
  -H "Content-Type: application/json" \
  -d @data/sample_customer.json
```

---

### Option 2: Always-On VM (Simple but Limited)

**Best for**: Demos, testing, small-scale production

**Costs**: $0/month (free tier VM)

#### Setup Steps

1. **Create VM Instance**
```bash
# Login to OCI Console
# Compute > Instances > Create Instance

# Configuration:
# - Name: churn-prediction-vm
# - Image: Canonical Ubuntu 22.04
# - Shape: VM.Standard.E2.1.Micro (Always Free)
# - Network: Create/select VCN with public subnet
# - Add SSH key: Use existing or create new

# Or via CLI:
oci compute instance launch \
  --availability-domain <AD> \
  --compartment-id <compartment-id> \
  --shape VM.Standard.E2.1.Micro \
  --display-name churn-prediction-vm \
  --image-id <ubuntu-22.04-image-id> \
  --subnet-id <subnet-id> \
  --ssh-authorized-keys-file ~/.ssh/id_rsa.pub
```

2. **Setup VM**
```bash
# SSH into VM
ssh ubuntu@<public-ip>

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv -y

# Create project directory
mkdir -p ~/churn-prediction
cd ~/churn-prediction

# Copy project files (use scp or git)
# From local machine:
scp -r predictive-analytics-pipeline ubuntu@<public-ip>:~/
```

3. **Install Python Environment**
```bash
# On VM
cd ~/predictive-analytics-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model (optional - or copy pre-trained from local)
python scripts/generate_sample_data.py
python src/train_pipeline.py
```

4. **Run API**
```bash
# Start API
python src/api.py

# Or use production server (gunicorn)
pip install gunicorn

gunicorn -w 2 -b 0.0.0.0:5000 src.api:app
```

5. **Configure Security List (Firewall)**
```bash
# In OCI Console:
# Networking > Virtual Cloud Networks > Your VCN > Security Lists
# Add Ingress Rule:
#   Source: 0.0.0.0/0
#   Destination Port: 5000
#   Protocol: TCP
```

6. **Setup as System Service (Keep Running)**
```bash
# Create systemd service
sudo nano /etc/systemd/system/churn-api.service
```

Add:
```ini
[Unit]
Description=Churn Prediction API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/predictive-analytics-pipeline
Environment="PATH=/home/ubuntu/predictive-analytics-pipeline/venv/bin"
ExecStart=/home/ubuntu/predictive-analytics-pipeline/venv/bin/gunicorn -w 2 -b 0.0.0.0:5000 src.api:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable churn-api
sudo systemctl start churn-api
sudo systemctl status churn-api

# Check logs
sudo journalctl -u churn-api -f
```

---

### Option 3: Hybrid (Train Locally, Store in OCI)

**Best for**: Portfolio demonstration with cloud integration

**Costs**: $0/month (uses Object Storage only)

#### Setup Steps

1. **Configure OCI CLI** (already done on your machine)
```bash
oci setup config
```

2. **Create Object Storage Bucket**
```bash
# Via CLI
oci os bucket create \
  --compartment-id <compartment-id> \
  --name churn-models

# Or via Console:
# Storage > Buckets > Create Bucket
```

3. **Upload Model Artifacts**
```bash
# Train locally
python scripts/generate_sample_data.py
python src/train_pipeline.py

# Upload to OCI
oci os object put \
  --bucket-name churn-models \
  --file models/churn_model.pkl \
  --name churn_model.pkl

oci os object put \
  --bucket-name churn-models \
  --file models/feature_engineer.pkl \
  --name feature_engineer.pkl

oci os object put \
  --bucket-name churn-models \
  --file models/model_metadata.json \
  --name model_metadata.json
```

4. **Download for Inference**
```bash
# On any machine with OCI CLI
oci os object get \
  --bucket-name churn-models \
  --file models/churn_model.pkl \
  --name churn_model.pkl

# Run API locally or on free tier VM
python src/api.py
```

---

## Cost Optimization Tips

### 1. Use Arm-Based Compute (Best Value)
```bash
# Arm instances provide 4x cores + 24GB memory (vs 2x AMD with 1GB)
# Shape: VM.Standard.A1.Flex
# Perfect for ML workloads
```

### 2. Schedule VM Start/Stop
```bash
# Stop VM when not in use (save on Block Storage access)
oci compute instance action --action STOP --instance-id <instance-id>

# Start when needed
oci compute instance action --action START --instance-id <instance-id>

# Automate with cron:
# Stop at 6 PM
0 18 * * * oci compute instance action --action STOP --instance-id <id>
# Start at 8 AM
0 8 * * * oci compute instance action --action START --instance-id <id>
```

### 3. Use Autonomous Database for Storage (Optional)
```bash
# Instead of PostgreSQL on VM, use free Oracle ATP
# Benefits:
# - Auto-scaling
# - Auto-backups
# - No maintenance
# - 20 GB free storage
```

---

## Monitoring & Alerts

### Setup OCI Monitoring (Free)
```bash
# Console: Observability & Management > Monitoring

# Create Alarm for:
# - High CPU usage
# - Low memory
# - High API latency
# - Error rate thresholds

# Example: Email alert if CPU > 80%
oci monitoring alarm create \
  --compartment-id <compartment-id> \
  --display-name "High CPU Alert" \
  --metric-compartment-id <compartment-id> \
  --namespace oci_computeagent \
  --query-text "CpuUtilization[1m].mean() > 80" \
  --severity "CRITICAL" \
  --destinations '["<topic-ocid>"]'
```

---

## Security Best Practices

1. **Use VCN Security Lists**
   - Only allow ports 22 (SSH), 5000 (API)
   - Restrict SSH to your IP only

2. **API Authentication** (Production)
```python
# Add to api.py
from functools import wraps
from flask import request

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... existing code
```

3. **HTTPS with Let's Encrypt** (Free SSL)
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com
```

---

## Troubleshooting

### VM Won't Start
```bash
# Check service limits
oci limits resource-availability get \
  --compartment-id <compartment-id> \
  --service-name compute

# Ensure you're using free tier shapes:
# - VM.Standard.E2.1.Micro (AMD)
# - VM.Standard.A1.Flex (Arm)
```

### Out of Memory
```bash
# Add swap space on VM
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### API Slow Response
```bash
# Optimize model loading
# Cache model in memory (already done in api.py)
# Use lighter model (reduce n_estimators in Random Forest)
# Enable model compression:
model = joblib.load('churn_model.pkl', mmap_mode='r')
```

---

## Next Steps

1. **Deploy using your preferred option** (Functions recommended)
2. **Test API with sample data**
3. **Add monitoring and alerts**
4. **Document deployment in your ePortfolio**
5. **Consider CI/CD with GitHub Actions** (future enhancement)

---

## Support Resources

- OCI Documentation: https://docs.oracle.com/en-us/iaas/
- OCI Free Tier: https://www.oracle.com/cloud/free/
- OCI CLI Reference: https://docs.oracle.com/en-us/iaas/tools/oci-cli/
- Community Forum: https://cloudcustomerconnect.oracle.com/

---

**Pro Tip**: For your ePortfolio, document your deployment with screenshots showing:
- ✅ OCI Console with running resources
- ✅ API endpoint returning predictions
- ✅ Cost analysis showing $0 charges
- ✅ Architecture diagram
- ✅ Monitoring dashboards

This demonstrates real-world cloud deployment skills!
