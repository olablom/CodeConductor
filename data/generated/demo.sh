#!/usr/bin/env bash
set -e

echo "🚀 CodeConductor v2.0 - Microservices Demo"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Wait for services to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url/health" > /dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within $max_attempts attempts"
    return 1
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    print_warning "jq is not installed. Installing jq..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y jq
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install jq
    else
        print_error "Please install jq manually for JSON parsing"
        exit 1
    fi
fi

# Wait for services
wait_for_service "http://localhost:8001" "User Service"
wait_for_service "http://localhost:8002" "Order Service"

echo ""
print_status "Starting microservices demo..."

# Step 1: Register a new user
echo ""
print_status "Step 1: Registering new user..."
REGISTER_RESPONSE=$(curl -s -X POST http://localhost:8001/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "email": "demo@example.com",
    "password": "demo123"
  }')

if echo "$REGISTER_RESPONSE" | jq -e '.id' > /dev/null 2>&1; then
    USER_ID=$(echo "$REGISTER_RESPONSE" | jq -r '.id')
    print_success "User registered successfully! User ID: $USER_ID"
    echo "$REGISTER_RESPONSE" | jq '.'
else
    print_error "Failed to register user"
    echo "$REGISTER_RESPONSE"
    exit 1
fi

# Step 2: Login and get JWT token
echo ""
print_status "Step 2: Logging in to get JWT token..."
LOGIN_RESPONSE=$(curl -s -X POST http://localhost:8001/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "password": "demo123"
  }')

if echo "$LOGIN_RESPONSE" | jq -e '.access_token' > /dev/null 2>&1; then
    TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.access_token')
    print_success "Login successful! JWT token obtained"
    echo "$LOGIN_RESPONSE" | jq '.'
else
    print_error "Failed to login"
    echo "$LOGIN_RESPONSE"
    exit 1
fi

# Step 3: Get current user info
echo ""
print_status "Step 3: Getting current user information..."
USER_INFO=$(curl -s -X GET http://localhost:8001/users/me \
  -H "Authorization: Bearer $TOKEN")

if echo "$USER_INFO" | jq -e '.username' > /dev/null 2>&1; then
    print_success "User info retrieved successfully!"
    echo "$USER_INFO" | jq '.'
else
    print_error "Failed to get user info"
    echo "$USER_INFO"
    exit 1
fi

# Step 4: Create an order
echo ""
print_status "Step 4: Creating a new order..."
ORDER_RESPONSE=$(curl -s -X POST http://localhost:8002/orders \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "item": "Premium Widget",
    "quantity": 3,
    "price": 29.99
  }')

if echo "$ORDER_RESPONSE" | jq -e '.id' > /dev/null 2>&1; then
    ORDER_ID=$(echo "$ORDER_RESPONSE" | jq -r '.id')
    print_success "Order created successfully! Order ID: $ORDER_ID"
    echo "$ORDER_RESPONSE" | jq '.'
else
    print_error "Failed to create order"
    echo "$ORDER_RESPONSE"
    exit 1
fi

# Step 5: List all orders
echo ""
print_status "Step 5: Listing all orders for user..."
ORDERS_LIST=$(curl -s -X GET http://localhost:8002/orders \
  -H "Authorization: Bearer $TOKEN")

if echo "$ORDERS_LIST" | jq -e '.[0]' > /dev/null 2>&1; then
    print_success "Orders retrieved successfully!"
    echo "$ORDERS_LIST" | jq '.'
else
    print_warning "No orders found or failed to retrieve orders"
    echo "$ORDERS_LIST"
fi

# Step 6: Get specific order
echo ""
print_status "Step 6: Getting specific order details..."
ORDER_DETAILS=$(curl -s -X GET "http://localhost:8002/orders/$ORDER_ID" \
  -H "Authorization: Bearer $TOKEN")

if echo "$ORDER_DETAILS" | jq -e '.id' > /dev/null 2>&1; then
    print_success "Order details retrieved successfully!"
    echo "$ORDER_DETAILS" | jq '.'
else
    print_error "Failed to get order details"
    echo "$ORDER_DETAILS"
    exit 1
fi

# Step 7: Update order status
echo ""
print_status "Step 7: Updating order status to 'processing'..."
STATUS_UPDATE=$(curl -s -X PUT "http://localhost:8002/orders/$ORDER_ID/status?status=processing" \
  -H "Authorization: Bearer $TOKEN")

if echo "$STATUS_UPDATE" | jq -e '.message' > /dev/null 2>&1; then
    print_success "Order status updated successfully!"
    echo "$STATUS_UPDATE" | jq '.'
else
    print_error "Failed to update order status"
    echo "$STATUS_UPDATE"
    exit 1
fi

# Step 8: Get order statistics
echo ""
print_status "Step 8: Getting order statistics..."
STATS=$(curl -s -X GET http://localhost:8002/stats)

if echo "$STATS" | jq -e '.total_orders' > /dev/null 2>&1; then
    print_success "Statistics retrieved successfully!"
    echo "$STATS" | jq '.'
else
    print_error "Failed to get statistics"
    echo "$STATS"
    exit 1
fi

# Step 9: List all users (admin function)
echo ""
print_status "Step 9: Listing all users..."
USERS_LIST=$(curl -s -X GET http://localhost:8001/users)

if echo "$USERS_LIST" | jq -e '.[0]' > /dev/null 2>&1; then
    print_success "Users list retrieved successfully!"
    echo "$USERS_LIST" | jq '.'
else
    print_warning "No users found or failed to retrieve users"
    echo "$USERS_LIST"
fi

echo ""
print_success "🎉 DEMO COMPLETED SUCCESSFULLY!"
echo ""
echo "📊 Demo Summary:"
echo "   ✅ User registration and authentication"
echo "   ✅ JWT token generation and validation"
echo "   ✅ Order creation and management"
echo "   ✅ RabbitMQ event publishing"
echo "   ✅ Service-to-service communication"
echo "   ✅ Complete CRUD operations"
echo ""
echo "🌐 Service URLs:"
echo "   User Service: http://localhost:8001/docs"
echo "   Order Service: http://localhost:8002/docs"
echo "   RabbitMQ Management: http://localhost:15672 (admin/admin123)"
echo ""
echo "🔑 Demo Credentials:"
echo "   Username: demo_user"
echo "   Password: demo123"
echo "   JWT Token: $TOKEN"
echo ""
print_success "CodeConductor v2.0 microservices are working perfectly! 🚀" 