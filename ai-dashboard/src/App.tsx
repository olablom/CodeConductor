import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  Container,
  Typography,
  AppBar,
  Toolbar,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Psychology as PsychologyIcon,
} from '@mui/icons-material';
import AIDecisionPanel from './components/AIDecisionPanel';
import PerformanceChart from './components/PerformanceChart';
import GPUMonitor from './components/GPUMonitor';
import SystemStatus from './components/SystemStatus';

// Dark theme for AI dashboard
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00d4ff',
    },
    secondary: {
      main: '#ff6b35',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
  },
});

interface AIDecision {
  id: string;
  timestamp: string;
  selected_arm: string;
  confidence: number;
  gpu_used: boolean;
  inference_time_ms: number;
  exploration: boolean;
}

interface GPUStats {
  gpu_available: boolean;
  device: string;
  gpu_memory_gb: number;
  gpu_memory_used_gb: number;
  gpu_memory_free_gb: number;
}

function App() {
  const [isBeastMode, setIsBeastMode] = useState(true);
  const [aiDecisions, setAiDecisions] = useState<AIDecision[]>([]);
  const [gpuStats, setGpuStats] = useState<GPUStats | null>(null);
  const [systemStatus, setSystemStatus] = useState({
    gpuService: false,
    prometheus: false,
    grafana: false,
  });
  const [error, setError] = useState<string | null>(null);

  // Check system status
  useEffect(() => {
    const checkSystemStatus = async () => {
      try {
        // Check GPU service
        const gpuResponse = await fetch('http://localhost:8007/health');
        setSystemStatus(prev => ({ ...prev, gpuService: gpuResponse.ok }));
        
        if (gpuResponse.ok) {
          const gpuData = await gpuResponse.json();
          setGpuStats(gpuData);
        }

        // Check Prometheus
        const promResponse = await fetch('http://localhost:9090/-/healthy');
        setSystemStatus(prev => ({ ...prev, prometheus: promResponse.ok }));

        // Check Grafana
        const grafanaResponse = await fetch('http://localhost:3000/api/health');
        setSystemStatus(prev => ({ ...prev, grafana: grafanaResponse.ok }));

      } catch (err) {
        setError('Failed to check system status');
      }
    };

    checkSystemStatus();
    const interval = setInterval(checkSystemStatus, 10000); // Check every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Generate AI decisions
  useEffect(() => {
    const generateAIDecision = async () => {
      try {
        const response = await fetch('http://localhost:8007/gpu/bandits/choose', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            arms: ['conservative_strategy', 'experimental_strategy', 'hybrid_approach'],
            features: [0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.9, 0.6, 0.7, 0.8],
            epsilon: 0.1,
          }),
        });

        if (response.ok) {
          const decision = await response.json();
          const newDecision: AIDecision = {
            id: Date.now().toString(),
            timestamp: new Date().toISOString(),
            selected_arm: decision.selected_arm,
            confidence: decision.confidence,
            gpu_used: decision.gpu_used,
            inference_time_ms: decision.inference_time_ms,
            exploration: decision.exploration,
          };

          setAiDecisions(prev => [newDecision, ...prev.slice(0, 9)]); // Keep last 10 decisions
        }
      } catch (err) {
        console.error('Failed to generate AI decision:', err);
      }
    };

    // Generate decision every 3 seconds
    const interval = setInterval(generateAIDecision, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        {/* Header */}
        <AppBar position="static" sx={{ background: 'linear-gradient(45deg, #00d4ff 30%, #ff6b35 90%)' }}>
          <Toolbar>
            <PsychologyIcon sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              CodeConductor AI Dashboard
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={isBeastMode}
                  onChange={(e) => setIsBeastMode(e.target.checked)}
                  sx={{
                    '& .MuiSwitch-switchBase.Mui-checked': {
                      color: '#00d4ff',
                    },
                    '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                      backgroundColor: '#00d4ff',
                    },
                  }}
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Chip
                    icon={isBeastMode ? <MemoryIcon /> : <SpeedIcon />}
                    label={isBeastMode ? 'BEAST MODE' : 'LIGHT MODE'}
                    color={isBeastMode ? 'primary' : 'secondary'}
                    size="small"
                    sx={{ ml: 1 }}
                  />
                </Box>
              }
            />
          </Toolbar>
        </AppBar>

        {/* Main Content */}
        <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
          {/* Error Alert */}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {/* System Status */}
          <SystemStatus status={systemStatus} />

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* GPU Monitor and Performance Chart Row */}
            <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
              <Box sx={{ flex: '1 1 400px', minWidth: 0 }}>
                <GPUMonitor gpuStats={gpuStats} isBeastMode={isBeastMode} />
              </Box>
              <Box sx={{ flex: '2 1 800px', minWidth: 0 }}>
                <PerformanceChart aiDecisions={aiDecisions} isBeastMode={isBeastMode} />
              </Box>
            </Box>

            {/* AI Decision Panel */}
            <Box>
              <AIDecisionPanel decisions={aiDecisions} isBeastMode={isBeastMode} />
            </Box>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
