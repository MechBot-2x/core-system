import React, { useState, useRef } from 'react';
import { Upload, FileText, Download, Plus, Trash2, Edit, Eye, Copy, Check } from 'lucide-react';

const ReadmeIntegrator = () => {
  const [mainReadme, setMainReadme] = useState('');
  const [subFiles, setSubFiles] = useState([]);
  const [integratedContent, setIntegratedContent] = useState('');
  const [activeTab, setActiveTab] = useState('upload');
  const [copied, setCopied] = useState(false);
  const fileInputRef = useRef(null);

  // Cargar el contenido inicial del README de Neural Nexus
  React.useEffect(() => {
    const initialContent = `# 🧠 Neural Nexus - Core System

[![Build Status](https://github.com/mechmind-dwv/core-system/actions/workflows/build.yml/badge.svg)](https://github.com/mechmind-dwv/core-system/actions)
[![Edge Latency](https://img.shields.io/badge/edge_latency-<2ms-green)](https://neural-nexus.dev/benchmarks)
[![Energy Efficiency](https://img.shields.io/badge/power_consumption-<5W-brightgreen)](https://neural-nexus.dev/efficiency)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/neural-nexus?color=7289da&label=discord)](https://discord.gg/neural-nexus)

> **Plataforma de IA Distribuida para Edge Computing** que combina **inferencia descentralizada**, **eficiencia energética** y **aprendizaje federado** con procesamiento **neuromorphic**. 🚀

## 🎯 ¿Qué es Neural Nexus?

Neural Nexus revoluciona el edge computing al distribuir la inteligencia artificial directamente en los dispositivos, eliminando la dependencia de la nube y garantizando:

- ⚡ **Ultra-baja latencia** (< 2ms)
- 🔋 **Eficiencia energética** (< 5W por nodo)  
- 🛡️ **Privacidad por diseño** (datos nunca salen del edge)
- 🌐 **Escalabilidad masiva** (miles de nodos)
- 🧠 **Aprendizaje continuo** (federated learning)

<!-- INTEGRATION_POINT:architecture -->
<!-- INTEGRATION_POINT:quick-start -->
<!-- INTEGRATION_POINT:project-structure -->
<!-- INTEGRATION_POINT:tech-stack -->
<!-- INTEGRATION_POINT:use-cases -->
<!-- INTEGRATION_POINT:benchmarks -->
<!-- INTEGRATION_POINT:deployment -->
<!-- INTEGRATION_POINT:development -->
<!-- INTEGRATION_POINT:security -->
<!-- INTEGRATION_POINT:roadmap -->
<!-- INTEGRATION_POINT:contributing -->
<!-- INTEGRATION_POINT:support -->
<!-- INTEGRATION_POINT:license -->`;
    
    setMainReadme(initialContent);
    setIntegratedContent(initialContent);
  }, []);

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    
    for (const file of files) {
      const content = await file.text();
      const newFile = {
        id: Date.now() + Math.random(),
        name: file.name,
        content: content,
        section: extractSectionName(file.name),
        integrationPoint: generateIntegrationPoint(file.name)
      };
      
      setSubFiles(prev => [...prev, newFile]);
    }
  };

  const extractSectionName = (filename) => {
    // Extraer nombre de sección basado en el nombre del archivo
    const name = filename.replace(/\.(md|txt)$/i, '');
    return name
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
  };

  const generateIntegrationPoint = (filename) => {
    // Generar punto de integración basado en el nombre del archivo
    return filename
      .replace(/\.(md|txt)$/i, '')
      .toLowerCase()
      .replace(/[^a-z0-9]/g, '-');
  };

  const addManualSection = () => {
    const newFile = {
      id: Date.now(),
      name: 'nueva-seccion.md',
      content: '## Nueva Sección\n\nContenido de la nueva sección...',
      section: 'Nueva Sección',
      integrationPoint: 'nueva-seccion'
    };
    
    setSubFiles(prev => [...prev, newFile]);
  };

  const updateSubFile = (id, field, value) => {
    setSubFiles(prev => prev.map(file => 
      file.id === id ? { ...file, [field]: value } : file
    ));
  };

  const deleteSubFile = (id) => {
    setSubFiles(prev => prev.filter(file => file.id !== id));
  };

  const integrateFiles = () => {
    let integrated = mainReadme;
    
    subFiles.forEach(file => {
      const integrationPoint = `<!-- INTEGRATION_POINT:${file.integrationPoint} -->`;
      
      if (integrated.includes(integrationPoint)) {
        // Reemplazar el punto de integración con el contenido
        integrated = integrated.replace(
          integrationPoint,
          `${file.content}\n\n${integrationPoint}`
        );
      } else {
        // Si no existe el punto de integración, agregar al final
        integrated += `\n\n${file.content}`;
      }
    });
    
    setIntegratedContent(integrated);
    setActiveTab('preview');
  };

  const downloadIntegratedFile = () => {
    const blob = new Blob([integratedContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'README-integrated.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(integratedContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Error copying to clipboard:', err);
    }
  };

  const generateTemplate = () => {
    const template = `# Proyecto

## Descripción
<!-- INTEGRATION_POINT:description -->

## Instalación
<!-- INTEGRATION_POINT:installation -->

## Uso
<!-- INTEGRATION_POINT:usage -->

## API
<!-- INTEGRATION_POINT:api -->

## Contribuir
<!-- INTEGRATION_POINT:contributing -->

## Licencia
<!-- INTEGRATION_POINT:license -->`;

    setMainReadme(template);
    setIntegratedContent(template);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2 flex items-center gap-2">
          <FileText className="text-blue-600" size={32} />
          Integrador de Archivos README
        </h1>
        <p className="text-gray-600">
          Herramienta para integrar múltiples archivos README y crear documentación unificada
        </p>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow-lg mb-6">
        <div className="flex border-b">
          {[
            { id: 'upload', label: 'Subir Archivos', icon: Upload },
            { id: 'manage', label: 'Gestionar Secciones', icon: Edit },
            { id: 'preview', label: 'Vista Previa', icon: Eye }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-6 py-3 font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              <tab.icon size={20} />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {activeTab === 'upload' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold mb-4">README Principal</h3>
                <div className="flex gap-4 mb-4">
                  <button
                    onClick={generateTemplate}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Generar Template
                  </button>
                </div>
                <textarea
                  value={mainReadme}
                  onChange={(e) => setMainReadme(e.target.value)}
                  className="w-full h-40 p-4 border border-gray-300 rounded-lg font-mono text-sm"
                  placeholder="Pega aquí tu README principal o usa el template..."
                />
              </div>

              <div>
                <h3 className="text-xl font-semibold mb-4">Subir Archivos Adicionales</h3>
                <div 
                  className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={48} className="mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600 mb-2">
                    Haz clic para subir archivos README o suéltalos aquí
                  </p>
                  <p className="text-sm text-gray-500">
                    Soporta archivos .md, .txt
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".md,.txt"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </div>

                <button
                  onClick={addManualSection}
                  className="mt-4 flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  <Plus size={20} />
                  Añadir Sección Manual
                </button>
              </div>
            </div>
          )}

          {activeTab === 'manage' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h3 className="text-xl font-semibold">Secciones a Integrar</h3>
                <button
                  onClick={integrateFiles}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Integrar Archivos
                </button>
              </div>

              {subFiles.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No hay archivos para integrar. Sube algunos archivos en la pestaña anterior.
                </div>
              ) : (
                <div className="space-y-4">
                  {subFiles.map(file => (
                    <div key={file.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex justify-between items-start mb-4">
                        <div className="flex-1 space-y-2">
                          <input
                            type="text"
                            value={file.name}
                            onChange={(e) => updateSubFile(file.id, 'name', e.target.value)}
                            className="text-lg font-semibold bg-transparent border-b border-transparent hover:border-gray-300 focus:border-blue-500 outline-none"
                          />
                          <input
                            type="text"
                            value={file.integrationPoint}
                            onChange={(e) => updateSubFile(file.id, 'integrationPoint', e.target.value)}
                            className="text-sm text-gray-600 bg-gray-50 px-2 py-1 rounded border"
                            placeholder="punto-de-integracion"
                          />
                        </div>
                        <button
                          onClick={() => deleteSubFile(file.id)}
                          className="text-red-600 hover:text-red-800 p-1"
                        >
                          <Trash2 size={20} />
                        </button>
                      </div>
                      <textarea
                        value={file.content}
                        onChange={(e) => updateSubFile(file.id, 'content', e.target.value)}
                        className="w-full h-32 p-3 border border-gray-300 rounded-lg font-mono text-sm"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'preview' && (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-xl font-semibold">README Integrado</h3>
                <div className="flex gap-2">
                  <button
                    onClick={copyToClipboard}
                    className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    {copied ? <Check size={20} /> : <Copy size={20} />}
                    {copied ? 'Copiado!' : 'Copiar'}
                  </button>
                  <button
                    onClick={downloadIntegratedFile}
                    className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    <Download size={20} />
                    Descargar
                  </button>
                </div>
              </div>
              
              <div className="border border-gray-300 rounded-lg">
                <div className="bg-gray-100 px-4 py-2 border-b border-gray-300 text-sm text-gray-600">
                  README-integrated.md
                </div>
                <div className="p-4 max-h-96 overflow-y-auto">
                  <pre className="whitespace-pre-wrap font-mono text-sm text-gray-800">
                    {integratedContent}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Instrucciones */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h4 className="font-semibold text-blue-800 mb-2">💡 Cómo usar la herramienta:</h4>
        <ol className="list-decimal list-inside space-y-1 text-blue-700 text-sm">
          <li>En tu README principal, usa comentarios como <code className="bg-blue-100 px-1 rounded"><!-- INTEGRATION_POINT:nombre-seccion --></code></li>
          <li>Sube los archivos que quieres integrar o créalos manualmente</li>
          <li>Ajusta los puntos de integración para que coincidan con los comentarios</li>
          <li>Haz clic en "Integrar Archivos" para generar el README final</li>
          <li>Descarga o copia el resultado integrado</li>
        </ol>
      </div>
    </div>
  );
};

export default ReadmeIntegrator;
