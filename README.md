# Jackpot Teller AI App

A Python/Flask web application for AI-based lottery analysis.

## Directory Structure

- db/models.py: SQLAlchemy models for Powerball/MegaMillions
- scripts/import_csv.py: Import historical CSV data
- scripts/update_draws.py: Weekly updater to fetch new draws
- training/train_powerball.py: Train PyTorch model for Powerball
- training/train_megamillions.py: Train PyTorch model for Mega Millions
- app/app.py: The main Flask web server
- app/templates/: HTML templates
- requirements.txt: Dependencies

## Steps

1. Create DB:

```
python db/models.py

or

python3.11 db/models.py
```

2. Import CSV:

```
python scripts/import_csv.py

or

python3.11 -m scripts.import_csv
```

3a. Train: Downgrade numpy to 1.x if needed!

```
pip install --upgrade "numpy<2"

or

pip3.11 install --upgrade "numpy<2"
```

3b. Train: After successfully installing numpy... These commands will create ".pt" files in the root folder

```
python training/train_powerball.py
python training/train_megamillions.py

or

python3.11 -m training.train_powerball
python3.11 -m training.train_megamillions
```

4. Run App:

```
cd app python app.py

or

at the root level:
python3.11 -m app.app

Access at http://127.0.0.1:5000
```

5. Weekly, run:

```
python scripts/update_draws.py

or

python3.11 -m scripts.update_draws
```

Then optionally re-train models with new data.

# Helpful commands

````markdown
# Deployment and Troubleshooting Guide for "Jackpot Teller AI" App on AWS EKS

This guide provides the necessary commands and steps to set up, verify, and troubleshoot the Jackpot Teller AI application running on AWS EKS, with a custom domain, HTTPS configuration, and automatic execution via a CronJob.

---

## 1. AWS CLI & kubeconfig Setup

### a. Verify AWS CLI and Configure Credentials

Make sure you have the AWS CLI installed and configured:

```bash
aws --version
aws configure  # (if not already configured, enter AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, region, and output format)
```
````

### b. Update kubeconfig for Your EKS Cluster

This command configures the local `kubectl` to use the EKS cluster:

```bash
aws eks update-kubeconfig --region us-west-2 --name ai-jackpot-cluster
```

---

## 2. Check Your EKS Cluster Status

### a. Check Nodes

Ensure the worker nodes are healthy:

```bash
kubectl get nodes
```

### b. Check Deployments, Pods, and Services

List deployments:

```bash
kubectl get deployments
```

List pods:

```bash
kubectl get pods
```

List services:

```bash
kubectl get svc
```

### c. Check Ingress Status

View Ingress and verify its ADDRESS (the ALB DNS name):

```bash
kubectl get ingress ai-jackpot-ingress
```

Describe the ingress for detailed info and events:

```bash
kubectl describe ingress ai-jackpot-ingress
```

---

## 3. Inspect Logs & Events

### a. Check AWS Load Balancer Controller Logs

Ensure the controller is running without errors:

```bash
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller
```

### b. Check Cluster Events (Optional)

List recent events to spot potential issues:

```bash
kubectl get events --sort-by='.metadata.creationTimestamp'
```

---

## 4. Manage Deployment

### a. Apply Kubernetes Manifests (if you make changes)

Apply deployment, CronJob, and Ingress manifests (assuming they are in the `k8s/` folder):

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/cronjob.yaml
kubectl apply -f k8s/ingress.yaml
```

### b. Force a Rollout Restart (to ensure new image is pulled)

Restart deployment so that nodes pull the latest image:

```bash
kubectl rollout restart deployment ai-jackpot-generator
```

### c. Check Rollout Status

Verify that the new pods are running:

```bash
kubectl rollout status deployment ai-jackpot-generator
```

---

## 5. Verify the ALB in AWS Console

### a. View Load Balancers

1. Log in to the AWS Management Console.
2. Navigate to **EC2 > Load Balancers**.
3. Confirm that the Application Load Balancer (ALB) is active.
4. Check the **Listeners** tab for port **443** (HTTPS) and verify that your ACM certificate is attached.

### b. (Optional) Using AWS CLI to Check Listeners

If you know the ALB ARN, you can run:

```bash
aws elbv2 describe-listeners --load-balancer-arn <your-alb-arn>
```

Replace `<your-alb-arn>` with your ALB’s ARN.

---

## 6. Docker & ECR (Optional Checks)

### a. Verify Docker Image in ECR

List images in the ECR repository:

```bash
aws ecr describe-images --repository-name apps/jackpotteller --region us-west-2
```

---

## 7. General Troubleshooting

- **Clear Browser Cache / Incognito:**  
  Sometimes old content is cached in the browser.

- **DNS Check:**  
  Verify that the custom domain resolves to the ALB using:

  ```bash
  dig <Custom Domain, e.g. www.jackpotteller.com> +short
  ```

- **Check Security Groups:**  
  In the AWS console, ensure the ALB’s security groups allow inbound traffic on ports **80** and **443**.

---

## 8. Summary of Automatic Execution

- **Kubernetes CronJob:**  
  The CronJob (defined in `k8s/cronjob.yaml`) is scheduled to run at a specified time (converted to UTC or using a `timeZone` field if supported) to trigger `auto_update.py` script. This script scrapes new data, updates the database, and triggers model retraining.

- **CI/CD Pipeline:**  
  GitHub Actions workflows automatically build the Docker image, push it to AWS ECR, and update EKS cluster by applying the Kubernetes manifests (deployment, cronjob, ingress) whenever changes are pushed to the `master` branch.

- **Custom Domain & HTTPS:**  
  The custom domain (`www.jackpotteller.com`) is configured in Route 53 as an Alias record pointing to the ALB. The Ingress resource instructs the ALB to terminate SSL with the ACM certificate, ensuring secure (HTTPS) connections.
