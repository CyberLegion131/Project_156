import axios from 'axios';
import toast from 'react-hot-toast';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling (no auth required)
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error);
    
    if (error.response?.status === 500) {
      toast.error('Server error. Please try again later.');
    } else if (error.code === 'ECONNABORTED') {
      toast.error('Request timeout. Please check your connection.');
    } else if (error.response?.data?.detail) {
      toast.error(error.response.data.detail);
    } else {
      toast.error('An unexpected error occurred.');
    }
    
    return Promise.reject(error);
  }
);

// API functions
export const authAPI = {
  login: async (credentials) => {
    try {
      const response = await api.post('/login', credentials);
      return response.data;
    } catch (error) {
      throw error;
    }
  },
};

export const predictionAPI = {
  predict: async (patientData) => {
    try {
      const response = await api.post('/predict', patientData);
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  getHistory: async (patientId) => {
    try {
      const response = await api.get(`/history/${patientId}`);
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  getModelInfo: async () => {
    try {
      const response = await api.get('/model/info');
      return response.data;
    } catch (error) {
      throw error;
    }
  },
};

export const healthAPI = {
  checkHealth: async () => {
    try {
      const response = await axios.get('/health');
      return response.data;
    } catch (error) {
      throw error;
    }
  },
};

// Utility functions
export const validateConnection = async () => {
  try {
    await healthAPI.checkHealth();
    return true;
  } catch (error) {
    return false;
  }
};

// Export default api instance for custom requests
export default api;