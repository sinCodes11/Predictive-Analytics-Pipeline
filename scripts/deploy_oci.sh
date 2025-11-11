#!/bin/bash
#
# OCI Deployment Script for Churn Prediction Pipeline
# Deploys model artifacts to OCI Object Storage
#

set -e

echo "======================================================"
echo "OCI Deployment Script - Churn Prediction Pipeline"
echo "======================================================"
echo ""

# Configuration
BUCKET_NAME="churn-models"
NAMESPACE=$(oci os ns get --query 'data' --raw-output 2>/dev/null)

if [ -z "$NAMESPACE" ]; then
    echo "❌ Error: OCI CLI not configured or not authenticated"
    echo "   Run: oci setup config"
    exit 1
fi

echo "✅ OCI Namespace: $NAMESPACE"
echo ""

# Check if bucket exists
echo "📦 Checking for bucket: $BUCKET_NAME..."
BUCKET_EXISTS=$(oci os bucket get --bucket-name "$BUCKET_NAME" 2>/dev/null || echo "")

if [ -z "$BUCKET_EXISTS" ]; then
    echo "   Bucket not found. Creating..."

    # Get compartment ID (use root compartment)
    COMPARTMENT_ID=$(oci iam compartment list --query 'data[0].id' --raw-output)

    oci os bucket create \
        --compartment-id "$COMPARTMENT_ID" \
        --name "$BUCKET_NAME" \
        --public-access-type NoPublicAccess

    echo "   ✅ Bucket created: $BUCKET_NAME"
else
    echo "   ✅ Bucket exists: $BUCKET_NAME"
fi

echo ""

# Upload model artifacts
MODELS_DIR="../models"

if [ ! -d "$MODELS_DIR" ]; then
    echo "❌ Error: Models directory not found"
    echo "   Expected: $MODELS_DIR"
    echo "   Run: python src/train_pipeline.py"
    exit 1
fi

echo "📤 Uploading model artifacts..."

# Upload model
if [ -f "$MODELS_DIR/churn_model.pkl" ]; then
    oci os object put \
        --bucket-name "$BUCKET_NAME" \
        --file "$MODELS_DIR/churn_model.pkl" \
        --name "churn_model.pkl" \
        --force
    echo "   ✅ Uploaded: churn_model.pkl"
else
    echo "   ⚠️  Not found: churn_model.pkl"
fi

# Upload feature engineer
if [ -f "$MODELS_DIR/feature_engineer.pkl" ]; then
    oci os object put \
        --bucket-name "$BUCKET_NAME" \
        --file "$MODELS_DIR/feature_engineer.pkl" \
        --name "feature_engineer.pkl" \
        --force
    echo "   ✅ Uploaded: feature_engineer.pkl"
else
    echo "   ⚠️  Not found: feature_engineer.pkl"
fi

# Upload metadata
if [ -f "$MODELS_DIR/model_metadata.json" ]; then
    oci os object put \
        --bucket-name "$BUCKET_NAME" \
        --file "$MODELS_DIR/model_metadata.json" \
        --name "model_metadata.json" \
        --force
    echo "   ✅ Uploaded: model_metadata.json"
else
    echo "   ⚠️  Not found: model_metadata.json"
fi

echo ""
echo "======================================================"
echo "✅ Deployment Complete!"
echo "======================================================"
echo ""
echo "View objects:"
echo "  oci os object list --bucket-name $BUCKET_NAME"
echo ""
echo "Download model:"
echo "  oci os object get --bucket-name $BUCKET_NAME \\"
echo "    --name churn_model.pkl --file churn_model.pkl"
echo ""
echo "Next steps:"
echo "  1. Deploy API to OCI Compute or Functions"
echo "  2. Configure model download in API startup"
echo "  3. See docs/OCI_SETUP.md for detailed instructions"
echo ""
