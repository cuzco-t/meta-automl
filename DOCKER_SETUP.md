# Guía de Migración a Docker

## Descripción General
Este proyecto ha sido migrado de un entorno virtual Python a una configuración con Docker, permitiendo una ejecución más consistente y portátil.

## Archivos Creados

### 1. **Dockerfile**
Define la imagen Docker del proyecto:
- Base: Python 3.11-slim (ligero)
- Instala dependencias del sistema requeridas
- Instala paquetes Python desde `requirements.txt`
- Copia el proyecto completo en `/app`
- Comando por defecto: ejecuta `main.py`

### 2. **requirements.txt**
Lista todas las dependencias Python necesarias:
- FLAML, scikit-learn, lightgbm, xgboost
- pandas, numpy, scipy
- Otras utilidades necesarias

### 3. **docker-compose.yml**
Orquestación de contenedores:
- Construye la imagen desde el `Dockerfile`
- Monta volúmenes para persistencia de datos
- Configura variables de entorno (`PYTHONUNBUFFERED=1`)
- Facilita ejecución con un comando simple

### 4. **.dockerignore**
Optimiza el tamaño de la imagen ignorando:
- Entorno virtual local (`.venv/`)
- Archivos de caché y compilados
- Archivos de IDE
- Archivos git

## Cómo Usar Docker

### Opción 1: Usar docker-compose (Recomendado)

**Construcción:**
```bash
docker-compose build
```

**Ejecutar:**
```bash
docker-compose run --rm meta-automl
```

**Ejecutar en modo interactivo:**
Edita `docker-compose.yml` y descomenta las líneas:
```yaml
stdin_open: true
tty: true
```
Luego ejecuta:
```bash
docker-compose run --rm meta-automl
```

### Opción 2: Usar Docker directamente

**Construcción:**
```bash
docker build -t meta-automl:latest .
```

**Ejecutar:**
```bash
docker run --rm -v %cd%/data:/app/data meta-automl:latest
```

En Linux/Mac:
```bash
docker run --rm -v $(pwd)/data:/app/data meta-automl:latest
```

### Opción 3: Ejecutar dentro del contenedor

```bash
docker run -it --rm meta-automl:latest /bin/bash
```

## Cambios Respecto al Entorno Virtual

| Aspecto | Antes (venv) | Ahora (Docker) |
|--------|--------------|----------------|
| Información | Dependencias en `.venv/` | Especificadas en `requirements.txt` |
| Distribución | Solo en local | Portátil en cualquier máquina |
| Variables de entorno | En `.env` local | En `docker-compose.yml` |
| Datos persistentes | En el host | Montados como volúmenes |
| Reproducibilidad | Puede variar por SO | Garantizada en contenedor |

## Variables de Entorno

Si necesitas variables de entorno, edita `docker-compose.yml`:
```yaml
environment:
  - TU_VARIABLE=valor
  - OTRA_VARIABLE=valor2
```

O crea un archivo `.env` en la raíz y modifica el compose:
```yaml
env_file:
  - .env
```

## Notas Importantes

1. **Datos**: Los datos en `./data/` se montan automáticamente en `/app/data` dentro del contenedor
2. **Puertos**: Si usas APIs (uvicorn), descomenta la sección `ports:` en `docker-compose.yml`
3. **Debugging**: Para debugging, usa la opción interactiva
4. **Rendimiento**: En Windows, el rendimiento puede ser menor. Considera usar WSL 2 para mejor performance

## Solución de Problemas

**Problema**: "Docker daemon no está corriendo"
- Solución: Inicia Docker Desktop

**Problema**: "Permission denied" en volúmenes
- Solución: Asegúrate de tener permisos sobre las carpetas

**Problema**: "Module not found"
- Solución: Verifica que todas las dependencias estén en `requirements.txt`

## Próximos Pasos Sugeridos

1. Testa localmente: `docker-compose up` o `docker-compose run --rm meta-automl`
2. Ajusta el `Dockerfile` si necesitas dependencias adicionales
3. Considera usar GitHub Actions para CI/CD con Docker
4. Publica la imagen a Docker Hub si lo necesitas compartir
