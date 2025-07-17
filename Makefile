init:
    pip install -r requirements.txt
    cargo build --release

deploy-edge:
    @echo "Selecciona dispositivo:"
    @echo "1. Jetson Xavier"
    @echo "2. Raspberry Pi 5"
    @read -p "Opción: " device; \
    case $$device in \
        1) cargo build --target=aarch64-unknown-linux-gnu ;; \
        2) cargo build --target=armv7-unknown-linux-gnueabihf ;; \
        *) echo "Opción no válida"; exit 1 ;; \
    esac
