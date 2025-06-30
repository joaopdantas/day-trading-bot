import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // Componente principal da aplicação
import './index.css'; // estilos globais (podes remover se não usares)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
