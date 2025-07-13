# Imagen base de Python
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de dependencias primero para aprovechar el cache de Docker
COPY requirements.txt ./

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código del proyecto
COPY . .

# Comando por defecto (puedes cambiarlo según el flujo principal)
CMD ["python", "main.py"] 