import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  CardHeader,
  Alert,
  CircularProgress,
  Box,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress
} from '@mui/material';
import {
  TrendingUp,
  Favorite,
  Thermostat,
  Warning,
  CheckCircle,
  Error
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  // State management
  const [vitalSigns, setVitalSigns] = useState({
    heart_rate: 75,
    spo2: 98,
    temperature: 37.0,
    hrv: 50,
    temp_trend: 0,
    patient_id: 'demo_patient'
  });
  
  const [prediction, setPrediction] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isInitializing, setIsInitializing] = useState(false);

  // Fetch system status on component mount
  useEffect(() => {
    fetchSystemStatus();
    fetchMetrics();
    fetchAlerts();
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/alerts`);
      setAlerts(response.data.alerts || []);
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    }
  };

  const initializeSystem = async () => {
    setIsInitializing(true);
    try {
      await axios.post(`${API_BASE_URL}/initialize`);
      setError(null);
      // Poll for completion
      const pollInterval = setInterval(async () => {
        const status = await axios.get(`${API_BASE_URL}/health`);
        if (status.data.is_trained) {
          clearInterval(pollInterval);
          setIsInitializing(false);
          fetchSystemStatus();
          fetchMetrics();
        }
      }, 5000);
    } catch (error) {
      setError('Failed to initialize system');
      setIsInitializing(false);
    }
  };

  const makePrediction = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, vitalSigns);
      setPrediction(response.data);
      fetchAlerts(); // Refresh alerts after prediction
      fetchSystemStatus(); // Update prediction count
    } catch (error) {
      setError(error.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setVitalSigns(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }));
  };

  const getRiskColor = (riskScore) => {
    if (riskScore >= 75) return 'error';
    if (riskScore >= 50) return 'warning';
    if (riskScore >= 25) return 'info';
    return 'success';
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'Normal':
        return <CheckCircle color="success" />;
      case 'Abnormal':
        return <Warning color="warning" />;
      default:
        return <Error color="error" />;
    }
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Typography variant="h3" component="h1" gutterBottom align="center" color="primary">
        🏥 Healthcare Monitoring System
      </Typography>
      
      {/* System Status */}
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" gutterBottom>
          System Status
        </Typography>
        {systemStatus && (
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    System Health
                  </Typography>
                  <Chip 
                    label={systemStatus.is_trained ? 'Trained' : 'Not Trained'} 
                    color={systemStatus.is_trained ? 'success' : 'error'}
                    icon={systemStatus.is_trained ? <CheckCircle /> : <Error />}
                  />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Predictions Made
                  </Typography>
                  <Typography variant="h6">
                    {systemStatus.total_predictions}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Total Alerts
                  </Typography>
                  <Typography variant="h6">
                    {systemStatus.total_alerts}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Uptime
                  </Typography>
                  <Typography variant="h6">
                    {systemStatus.uptime}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
        
        {!systemStatus?.is_trained && (
          <Box sx={{ mt: 2 }}>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={initializeSystem}
              disabled={isInitializing}
              startIcon={isInitializing ? <CircularProgress size={20} /> : null}
            >
              {isInitializing ? 'Initializing System...' : 'Initialize & Train Models'}
            </Button>
            {isInitializing && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                This may take 2-3 minutes. The system is training ML models...
              </Typography>
            )}
          </Box>
        )}
      </Paper>

      {/* Model Performance Metrics */}
      {metrics && (
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Typography variant="h5" gutterBottom>
            Model Performance
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Binary Classification
                  </Typography>
                  <Typography variant="h6">
                    {(metrics.binary_accuracy * 100).toFixed(1)}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={metrics.binary_accuracy * 100} 
                    color={metrics.binary_accuracy >= 0.94 ? 'success' : 'warning'}
                  />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Multi-class Classification
                  </Typography>
                  <Typography variant="h6">
                    {(metrics.multiclass_accuracy * 100).toFixed(1)}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={metrics.multiclass_accuracy * 100} 
                    color={metrics.multiclass_accuracy >= 0.85 ? 'success' : 'warning'}
                  />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Risk Prediction
                  </Typography>
                  <Typography variant="h6">
                    {(metrics.risk_accuracy * 100).toFixed(1)}%
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={metrics.risk_accuracy * 100} 
                    color={metrics.risk_accuracy >= 0.80 ? 'success' : 'warning'}
                  />
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      )}

      <Grid container spacing={4}>
        {/* Vital Signs Input */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Patient Vital Signs
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Heart Rate (bpm)"
                  type="number"
                  value={vitalSigns.heart_rate}
                  onChange={(e) => handleInputChange('heart_rate', e.target.value)}
                  InputProps={{
                    startAdornment: <Favorite color="error" sx={{ mr: 1 }} />
                  }}
                  inputProps={{ min: 30, max: 200 }}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="SpO2 (%)"
                  type="number"
                  value={vitalSigns.spo2}
                  onChange={(e) => handleInputChange('spo2', e.target.value)}
                  InputProps={{
                    startAdornment: <TrendingUp color="primary" sx={{ mr: 1 }} />
                  }}
                  inputProps={{ min: 70, max: 100 }}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Temperature (°C)"
                  type="number"
                  step="0.1"
                  value={vitalSigns.temperature}
                  onChange={(e) => handleInputChange('temperature', e.target.value)}
                  InputProps={{
                    startAdornment: <Thermostat color="warning" sx={{ mr: 1 }} />
                  }}
                  inputProps={{ min: 35, max: 42, step: 0.1 }}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Patient ID"
                  value={vitalSigns.patient_id}
                  onChange={(e) => handleInputChange('patient_id', e.target.value)}
                />
              </Grid>
            </Grid>
            
            <Box sx={{ mt: 3 }}>
              <Button
                variant="contained"
                color="primary"
                size="large"
                fullWidth
                onClick={makePrediction}
                disabled={loading || !systemStatus?.is_trained}
                startIcon={loading ? <CircularProgress size={20} /> : null}
              >
                {loading ? 'Analyzing...' : 'Analyze Vital Signs'}
              </Button>
            </Box>
            
            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </Paper>
        </Grid>

        {/* Prediction Results */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Health Analysis Results
            </Typography>
            
            {prediction ? (
              <Box>
                <Card sx={{ mb: 2 }}>
                  <CardHeader
                    avatar={getStatusIcon(prediction.binary_prediction)}
                    title={
                      <Typography variant="h6">
                        Status: {prediction.binary_prediction}
                      </Typography>
                    }
                    subheader={`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`}
                  />
                  <CardContent>
                    <Grid container spacing={2}>
                      <Grid item xs={12}>
                        <Typography variant="body1">
                          <strong>Detected Condition:</strong> {prediction.condition}
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="body1">
                          <strong>Risk Category:</strong> {prediction.risk_category}
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body1">
                            <strong>Risk Score:</strong>
                          </Typography>
                          <Chip 
                            label={`${prediction.risk_score.toFixed(0)}/100`}
                            color={getRiskColor(prediction.risk_score)}
                            variant="filled"
                          />
                        </Box>
                      </Grid>
                      <Grid item xs={12}>
                        <Alert 
                          severity={prediction.risk_score > 75 ? 'error' : 
                                   prediction.risk_score > 50 ? 'warning' : 'success'}
                        >
                          <strong>Recommended Action:</strong> {prediction.recommended_action}
                        </Alert>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Box>
            ) : (
              <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center', py: 4 }}>
                Enter vital signs and click "Analyze" to see health assessment
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Recent Alerts */}
      {alerts.length > 0 && (
        <Paper elevation={3} sx={{ p: 3, mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Recent Alerts
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Time</TableCell>
                  <TableCell>Patient ID</TableCell>
                  <TableCell>Alert Level</TableCell>
                  <TableCell>Condition</TableCell>
                  <TableCell>Risk Score</TableCell>
                  <TableCell>Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {alerts.slice(-10).reverse().map((alert, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </TableCell>
                    <TableCell>{alert.patient_id}</TableCell>
                    <TableCell>
                      <Chip 
                        label={alert.alert_level}
                        color={
                          alert.alert_level === 'CRITICAL' ? 'error' :
                          alert.alert_level === 'HIGH' ? 'warning' :
                          alert.alert_level === 'MEDIUM' ? 'info' : 'success'
                        }
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{alert.condition}</TableCell>
                    <TableCell>{alert.risk_score.toFixed(0)}</TableCell>
                    <TableCell>{alert.recommended_action}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}
    </Container>
  );
}

export default App;
