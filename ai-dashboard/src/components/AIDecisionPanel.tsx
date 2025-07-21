import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
} from '@mui/material';
import {
  Psychology as PsychologyIcon,
  Explore as ExploreIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';

interface AIDecision {
  id: string;
  timestamp: string;
  selected_arm: string;
  confidence: number;
  gpu_used: boolean;
  inference_time_ms: number;
  exploration: boolean;
}

interface AIDecisionPanelProps {
  decisions: AIDecision[];
  isBeastMode: boolean;
}

const AIDecisionPanel: React.FC<AIDecisionPanelProps> = ({ decisions, isBeastMode }) => {
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getArmColor = (arm: string) => {
    switch (arm) {
      case 'conservative_strategy':
        return 'success';
      case 'experimental_strategy':
        return 'warning';
      case 'hybrid_approach':
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Card sx={{ 
      background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
      border: `2px solid ${isBeastMode ? '#00d4ff' : '#ff6b35'}`,
    }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <PsychologyIcon sx={{ mr: 1, color: isBeastMode ? '#00d4ff' : '#ff6b35' }} />
          <Typography variant="h6" sx={{ color: isBeastMode ? '#00d4ff' : '#ff6b35' }}>
            Real-time AI Decisions
          </Typography>
          <Chip
            label={`${decisions.length} decisions`}
            size="small"
            color="primary"
            sx={{ ml: 'auto' }}
          />
        </Box>

        {decisions.length > 0 ? (
          <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ color: '#00d4ff', fontWeight: 'bold' }}>Time</TableCell>
                  <TableCell sx={{ color: '#00d4ff', fontWeight: 'bold' }}>Strategy</TableCell>
                  <TableCell sx={{ color: '#00d4ff', fontWeight: 'bold' }}>Confidence</TableCell>
                  <TableCell sx={{ color: '#00d4ff', fontWeight: 'bold' }}>Inference</TableCell>
                  <TableCell sx={{ color: '#00d4ff', fontWeight: 'bold' }}>Mode</TableCell>
                  <TableCell sx={{ color: '#00d4ff', fontWeight: 'bold' }}>Type</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {decisions.map((decision) => (
                  <TableRow key={decision.id} sx={{ '&:hover': { backgroundColor: 'rgba(0, 212, 255, 0.1)' } }}>
                    <TableCell sx={{ color: 'text.secondary' }}>
                      {formatTimestamp(decision.timestamp)}
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={decision.selected_arm.replace('_', ' ')}
                        size="small"
                        color={getArmColor(decision.selected_arm) as any}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          {(decision.confidence * 100).toFixed(1)}%
                        </Typography>
                        <Box
                          sx={{
                            width: 40,
                            height: 4,
                            backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            borderRadius: 2,
                            overflow: 'hidden',
                          }}
                        >
                          <Box
                            sx={{
                              width: `${decision.confidence * 100}%`,
                              height: '100%',
                              background: isBeastMode 
                                ? 'linear-gradient(90deg, #00d4ff 0%, #0099cc 100%)'
                                : 'linear-gradient(90deg, #ff6b35 0%, #cc5500 100%)',
                            }}
                          />
                        </Box>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {decision.gpu_used ? (
                          <MemoryIcon sx={{ color: '#00d4ff', mr: 0.5, fontSize: 16 }} />
                        ) : (
                          <SpeedIcon sx={{ color: '#ff6b35', mr: 0.5, fontSize: 16 }} />
                        )}
                        <Typography variant="body2">
                          {decision.inference_time_ms.toFixed(2)}ms
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={decision.gpu_used ? <MemoryIcon /> : <SpeedIcon />}
                        label={decision.gpu_used ? 'GPU' : 'CPU'}
                        size="small"
                        color={decision.gpu_used ? 'primary' : 'secondary'}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        icon={<ExploreIcon />}
                        label={decision.exploration ? 'Explore' : 'Exploit'}
                        size="small"
                        color={decision.exploration ? 'warning' : 'success'}
                        variant="outlined"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            height: 200,
            flexDirection: 'column',
          }}>
            <PsychologyIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>
              Waiting for AI decisions...
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Real-time decisions will appear here
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default AIDecisionPanel; 