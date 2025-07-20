# CodeConductor Microservices Architecture

## Overview

CodeConductor är uppdelat i följande mikrotjänster för skalbarhet och underhållbarhet:

## Service Architecture

### 1. Agent Service

- **Port**: 8001
- **Purpose**: Hanterar alla agent-relaterade operationer
- **Agents**: CodeGen, Architect, Review, Policy, Q-Learning
- **API**: REST/gRPC
- **Database**: Agent state, Q-tables, learning data

### 2. Orchestrator Service

- **Port**: 8002
- **Purpose**: Koordinerar anrop mellan agent-tjänsterna
- **Features**: Workflow management, state management, consensus logic
- **API**: REST/gRPC
- **Database**: Discussion history, consensus data

### 3. Data Service

- **Port**: 8003
- **Purpose**: Abstraktion över databasen
- **Features**: CRUD operations, data validation, caching
- **API**: REST/gRPC
- **Database**: All persistent data

### 4. Queue Service

- **Port**: 8004
- **Purpose**: Mellanlager för asynkrona meddelanden
- **Features**: Message routing, retry logic, dead letter queues
- **Technology**: RabbitMQ/Kafka
- **API**: AMQP/Kafka protocol

### 5. Auth/Approval Service

- **Port**: 8005
- **Purpose**: Hanterar mänsklig godkännandestatus och tokens
- **Features**: Human approval workflow, authentication, authorization
- **API**: REST
- **Database**: Approval history, user data

### 6. Gateway/API Service

- **Port**: 8000
- **Purpose**: Endpunkt mot externa klienter
- **Features**: API routing, rate limiting, authentication
- **API**: REST
- **Technology**: FastAPI + reverse proxy

## Communication Flow

```
Client → Gateway → [Agent/Orchestrator/Data/Auth] → Queue → [Async Processing]
```

## Development Setup

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up agent-service

# View logs
docker-compose logs -f
```

## API Documentation

- **Gateway**: http://localhost:8000/docs
- **Agent Service**: http://localhost:8001/docs
- **Orchestrator Service**: http://localhost:8002/docs
- **Data Service**: http://localhost:8003/docs
- **Auth Service**: http://localhost:8005/docs
