
# Sign To Speech Assistant

---

## Setup

- Install UV (inside a powershell window)

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

- Download the repo and put it in your desired folder

- Initiliaze the project

    ```powershell
    uv sync
    ```

    Will take some time if it's your first time since it downloads the flutter engine.

- Enter the python virtual env

    ```powerhsell
    .venv\Scripts\activate
    ```

---

## Run the project

- Download the flet app on your mobile phone

- In your terminal

    ```powershell
    uv run flet run --android
    ```

- Scan the QR code and test out the app from the flet application

Flet supports hot reloading by default so you don't have to run the project each time.
