# Camber Cloud para OrigenLM

Este repositorio se puede trasladar a Camber Cloud para realizar experimentos remotos. **Preparado** no significa **autorizado** ni **desplegado**.

## Seguridad

- No pegues claves de API en ChatGPT, terminales compartidas, capturas ni commits.
- Una clave compartida en un chat debe revocarse y sustituirse.
- **Preferido:** conecta ChatGPT con el MCP oficial de Camber usando OAuth; no requiere copiar la clave.
- No confundir este MCP (servicios de Camber) con un MCP que ejecute comandos dentro de GitHub Codespaces. Son servicios diferentes.

## Conectar Camber a ChatGPT con OAuth

Fuente oficial: https://docs.cambercloud.com/docs/camber-mcp/

1. Abre ChatGPT -> Ajustes -> Seguridad e inicio de sesión -> Modo desarrollador (si tu cuenta dispone de la función).
2. En la sección Plugins crea un plugin MCP remoto.
3. Nombre: `Camber Cloud`.
4. URL del servidor: `https://camber-mcp.cambercloud.com/mcp`.
5. Autenticación: OAuth.
6. Configuración OAuth avanzada: **User-Defined OAuth Client**.
7. Client ID **público** documentado por Camber para ChatGPT: `Nxc59XcXdNQUiI9V`.
8. Client Secret: **vacío**. Token endpoint auth method: `none`.
9. Confirma, inicia sesión en el sitio de Camber y aprueba la autorización.
10. Abre una nueva conversación con el plugin Camber activado.

Si la función de plugins MCP personalizados no aparece en ChatGPT, esta ruta no está disponible para esa cuenta; no improvises un enlace público ni des permisos a un servidor desconocido.

## Usar el CLI de Camber desde Codespaces

Documentación: https://docs.cambercloud.com/docs/camber-cli/installation/

Desde un Codespace correctamente configurado, Camber recomienda:

```bash
# Inspecciona el instalador antes de ejecutarlo.
curl -fsSL https://cli.cambercloud.com/install-v2.sh -o /tmp/camber-install.sh
less /tmp/camber-install.sh
bash /tmp/camber-install.sh
camber version
camber login
camber me
```

`camber login` requiere que el propietario de la cuenta complete la autenticación. No necesitas almacenar `CAMBER_API_KEY` si usas OAuth o login interactivo.

## Preparación segura del proyecto (sin subirlo)

```bash
python3 scripts/preparar_camber.py
# Opcional: incluir el checkpoint existente de prueba:
python3 scripts/preparar_camber.py --include-checkpoint
```

Genera `dist/OrigenLM-Camber.zip` con una lista permitida de archivos fuente, sin claves, carpetas `.git`, entornos virtuales ni archivos `.env`. **No envía nada a Camber**; la transferencia debe hacerse después de autorizar tu cuenta y comprobar cuotas/cargos.

## Diagnosticar Alpine vs Python/Codespaces

Si `gh codespace ssh` abre Alpine y `python3 chat.py` falla al importar `torch`, comprueba el entorno antes de instalar paquetes:

```bash
python3 scripts/diagnostico_entorno.py
```

OrigenLM espera imagen Debian Python 3.11 según `.devcontainer/devcontainer.json`. La presencia de Python en Alpine NO significa que PyTorch esté instalado. Usa el log de reconstrucción de Codespaces para averiguar si se inició un contenedor de recuperación o una imagen incorrecta.

Para ver los registros de Codespaces desde PowerShell:

```powershell
gh codespace logs -c scaling-waffle-jjv96796694cqv9w
```

El checkpoint `checkpoints/origen_latest.pt` está en la carpeta persistente `/workspaces`; no lo elimines.
