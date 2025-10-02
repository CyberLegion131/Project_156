import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import toast from 'react-hot-toast';
import { User, Lock, LogIn } from 'lucide-react';
import { authAPI } from '../api';

const Login = ({ onLogin }) => {
  const [isLoading, setIsLoading] = useState(false);
  const { register, handleSubmit, formState: { errors } } = useForm();

  const onSubmit = async (data) => {
    setIsLoading(true);
    try {
      const response = await authAPI.login(data);
      toast.success('Login successful!');
      onLogin(response.access_token);
    } catch (error) {
      console.error('Login failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container fade-in">
      <div className="login-header">
        <h2>🏥 DN Detection System</h2>
        <p>Sign in to access the diagnostic system</p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="login-form">
        <div className="form-group">
          <label htmlFor="username">
            <User size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
            Username
          </label>
          <input
            id="username"
            type="text"
            placeholder="Enter your username"
            {...register('username', { 
              required: 'Username is required',
              minLength: {
                value: 3,
                message: 'Username must be at least 3 characters'
              }
            })}
            className={errors.username ? 'error' : ''}
          />
          {errors.username && (
            <span className="error-message">{errors.username.message}</span>
          )}
        </div>

        <div className="form-group">
          <label htmlFor="password">
            <Lock size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
            Password
          </label>
          <input
            id="password"
            type="password"
            placeholder="Enter your password"
            {...register('password', { 
              required: 'Password is required',
              minLength: {
                value: 6,
                message: 'Password must be at least 6 characters'
              }
            })}
            className={errors.password ? 'error' : ''}
          />
          {errors.password && (
            <span className="error-message">{errors.password.message}</span>
          )}
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="btn btn-primary"
        >
          {isLoading ? (
            <>
              <div className="loading-spinner"></div>
              Signing in...
            </>
          ) : (
            <>
              <LogIn size={16} />
              Sign In
            </>
          )}
        </button>
      </form>

      <div style={{ marginTop: '2rem', padding: '1rem', background: '#f7fafc', borderRadius: '0.5rem', textAlign: 'center' }}>
        <p style={{ fontSize: '0.9rem', color: '#718096', marginBottom: '0.5rem' }}>
          Demo Credentials:
        </p>
        <p style={{ fontSize: '0.8rem', color: '#4a5568' }}>
          Username: <strong>admin</strong> | Password: <strong>admin123</strong>
        </p>
      </div>
    </div>
  );
};

export default Login;