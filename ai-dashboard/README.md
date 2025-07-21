# 🚀 CodeConductor AI Dashboard

## 🎯 Real-time AI Monitoring & Visualization

A professional React dashboard that provides real-time monitoring and visualization of AI algorithms running on the RTX 5090 GPU.

## ✨ Features

### 🔥 Real-time AI Monitoring

- **Live AI Decisions**: Stream neural bandit decisions every 3 seconds
- **GPU Performance**: Real-time RTX 5090 memory and performance metrics
- **System Status**: Monitor all CodeConductor services health
- **Beast vs Light Mode**: Toggle between GPU and CPU optimization modes

### 📊 Professional Visualizations

- **Performance Charts**: Interactive charts showing confidence and inference time
- **AI Decision Table**: Real-time table of all AI decisions with details
- **GPU Monitor**: Live GPU memory usage and performance indicators
- **System Health**: Service status monitoring with visual indicators

### 🎨 Modern UI/UX

- **Dark Theme**: Professional dark theme optimized for AI monitoring
- **Material-UI**: Modern React components with professional styling
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Live data updates without page refresh

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React App     │    │   GPU Service   │    │   MLOps Stack   │
│   (Frontend)    │◄──►│   (RTX 5090)    │◄──►│   (Monitoring)  │
│                 │    │                 │    │                 │
│ • Real-time UI  │    │ • Neural AI     │    │ • Prometheus    │
│ • Charts        │    │ • GPU Memory    │    │ • Grafana       │
│ • Controls      │    │ • Performance   │    │ • Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm
- CodeConductor GPU Service running on port 8007
- MLOps stack (Prometheus/Grafana) running

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

### Access Points

- **React Dashboard**: http://localhost:3000
- **GPU Service**: http://localhost:8007/health
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/codeconductor)

## 🎮 Usage

### Beast vs Light Mode Toggle

- **Beast Mode**: GPU-accelerated AI with RTX 5090
- **Light Mode**: CPU-optimized AI processing
- Toggle in the header to switch between modes

### Real-time Monitoring

- **AI Decisions**: Watch live neural bandit decisions
- **Performance**: Monitor inference time and confidence
- **GPU Stats**: Track RTX 5090 memory usage
- **System Health**: Monitor all service statuses

### Interactive Features

- **Hover Charts**: Detailed tooltips on performance data
- **Live Updates**: Real-time data streaming
- **Responsive Design**: Works on all screen sizes

## 🛠️ Technology Stack

### Frontend

- **React 18**: Modern React with hooks
- **TypeScript**: Type-safe development
- **Material-UI**: Professional UI components
- **Recharts**: Interactive data visualizations

### Backend Integration

- **REST API**: Communication with GPU service
- **Real-time Data**: Live AI decision streaming
- **Health Checks**: System status monitoring

### Styling

- **Dark Theme**: Professional AI monitoring theme
- **Gradients**: Modern gradient backgrounds
- **Responsive**: Mobile-first design
- **Animations**: Smooth transitions and effects

## 📊 Data Flow

1. **GPU Service** generates AI decisions every 3 seconds
2. **React App** fetches decisions via REST API
3. **Charts** update with real-time performance data
4. **UI Components** reflect current system status
5. **User Controls** allow mode switching and interaction

## 🎯 Portfolio Features

### Professional Showcase

- **Real-time AI**: Live demonstration of AI algorithms
- **GPU Integration**: RTX 5090 performance visualization
- **Modern UI**: Professional dashboard design
- **Full-stack**: Complete frontend to backend integration

### Technical Excellence

- **TypeScript**: Type-safe development
- **Modern React**: Hooks and functional components
- **Professional UI**: Material-UI with custom theming
- **Real-time Data**: Live streaming and updates

## 🔧 Development

### Project Structure

```
src/
├── components/
│   ├── AIDecisionPanel.tsx    # Real-time AI decisions table
│   ├── PerformanceChart.tsx   # Interactive performance charts
│   ├── GPUMonitor.tsx         # GPU statistics monitor
│   └── SystemStatus.tsx       # System health status
├── App.tsx                    # Main application component
└── index.tsx                  # Application entry point
```

### Key Components

- **App.tsx**: Main application with state management
- **AIDecisionPanel**: Real-time AI decisions display
- **PerformanceChart**: Interactive performance visualizations
- **GPUMonitor**: GPU statistics and monitoring
- **SystemStatus**: Service health monitoring

### State Management

- **useState**: Local component state
- **useEffect**: Side effects and data fetching
- **Real-time Updates**: Automatic data refresh
- **Error Handling**: Graceful error states

## 🎉 Success Metrics

### Technical Achievements

- ✅ **Real-time AI Monitoring**: Live streaming of AI decisions
- ✅ **GPU Integration**: RTX 5090 performance visualization
- ✅ **Professional UI**: Modern, responsive dashboard
- ✅ **Full-stack Integration**: Complete frontend to backend

### Portfolio Impact

- ✅ **Visual Demonstration**: Interactive AI showcase
- ✅ **Technical Depth**: Modern React with TypeScript
- ✅ **Professional Polish**: Enterprise-grade UI/UX
- ✅ **Real-time Capabilities**: Live data streaming

## 🚀 Future Enhancements

### Planned Features

- **WebSocket Integration**: Real-time bidirectional communication
- **Advanced Charts**: More sophisticated visualizations
- **User Authentication**: Secure access control
- **Mobile App**: React Native version

### Performance Optimizations

- **Virtual Scrolling**: Handle large datasets
- **Caching**: Optimize API calls
- **Lazy Loading**: Improve initial load time
- **PWA**: Progressive web app features

## 📝 License

This project is part of the CodeConductor AI platform demonstration.

---

**Built with ❤️ for AI Engineering Excellence**
