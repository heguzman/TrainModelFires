# Guía paso a paso para implementar DVC en tu proyecto

Esta guía te ayudará a implementar Data Version Control (DVC) en tu proyecto de machine learning para gestionar y versionar tus datos de manera eficiente.

---

## 1. Instalación de DVC

Instala DVC y el soporte para Google Drive:
```bash
pip install "dvc[gdrive]"
```

---

## 2. Inicializar DVC en el proyecto

Desde la raíz del proyecto:
```bash
dvc init
```
Esto creará la carpeta `.dvc/` y archivos de configuración.

---

## 3. Organización de la carpeta de datos

Recomendado:
- `data/raw/` — Datos originales sin procesar
- `data/processed/` — Datos listos para el modelado

Ejemplo de estructura:
```
data/
  raw/
    totalComplete.csv
    ImageComplete.csv
  processed/
```

---

## 4. Versionar archivos de datos con DVC

Agrega los archivos de datos al control de DVC:
```bash
dvc add data/raw/totalComplete.csv
dvc add data/raw/ImageComplete.csv
```
Esto generará archivos `.dvc` y actualizará `.gitignore` para evitar subir los datos a Git.

---

## 5. Configurar un remoto de DVC (Google Drive)

### 5.1. Agregar el remoto

```bash
dvc remote add -d gdrive_remote gdrive://<ID-de-tu-carpeta>
```
Ejemplo:
```bash
dvc remote add -d gdrive_remote gdrive://1eVTMtIpUlMEIRV7YwiHuokS61-7R-ELL
```

### 5.2. (Opcional) Usar credenciales propias de Google Cloud
Si Google bloquea el acceso, puedes crear un proyecto en Google Cloud, habilitar la API de Drive y obtener un Client ID y Secret. Luego:
```bash
dvc remote modify gdrive_remote gdrive_client_id 'tu-client-id'
dvc remote modify gdrive_remote gdrive_client_secret 'tu-client-secret'
```
Sigue la [guía oficial de DVC para Google Drive](https://dvc.org/doc/user-guide/setup-google-drive-remote#using-client-id-and-client-secret) para más detalles.

---

## 6. Guardar los cambios en Git

```bash
git add .dvc/config .gitignore data/raw/*.dvc
git commit -m "Configura DVC y versiona archivos de datos iniciales"
```

---

## 7. Subir los datos al remoto

```bash
dvc push
```
La primera vez, DVC te pedirá autenticarte con Google. Sigue el enlace y pega el código de autorización.

---

## 8. Descargar los datos en otra máquina o por otros usuarios

```bash
dvc pull
```
Esto descargará los datos versionados desde el remoto configurado.

---

## 9. Alternativa: Usar otro tipo de remoto

DVC soporta otros servicios como S3, Azure, Google Cloud Storage, SSH, WebDAV, etc. Consulta la [lista de remotos soportados](https://dvc.org/doc/command-reference/remote/add#supported-storage-types).

---

## 10. Documentar el flujo de datos

Crea o actualiza `data/README.md` para explicar el origen y uso de cada archivo de datos.

---

## 11. Recursos útiles
- [Documentación oficial de DVC](https://dvc.org/doc/)
- [Guía para Google Drive remoto](https://dvc.org/doc/user-guide/setup-google-drive-remote)
- [Lista de remotos soportados](https://dvc.org/doc/command-reference/remote/add#supported-storage-types) 