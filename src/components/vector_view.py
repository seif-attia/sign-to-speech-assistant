import flet as ft


class VectorView(ft.Container):
    """
    UI panel component that renders a scrollable text area for viewing dense 
    multidimensional vector outputs from MediaPipe.
    """

    def __init__(self, width: int = 300) -> None:
        # Monospace selectable text control to display landmark coordinate strings
        self.text_control = ft.Text(
            value="Waiting for MediaPipe landmarks...",
            color="#00FF00",
            size=12,
            font_family="monospace",
            selectable=True,
        )
        super().__init__(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "126-Dim Landmark Vector",
                        weight=ft.FontWeight.BOLD,
                        size=16,
                    ),
                    ft.Divider(height=1, color="#444444"),
                    # Scrollable Column wrapping text control to handle long vector strings
                    ft.Column(
                        controls=[self.text_control],
                        scroll=ft.ScrollMode.AUTO,
                        expand=True,  # Allows internal scroll area to stretch
                    ),
                ],
            ),
            width=width,
            expand=True,  # Stretches entire VectorView container inside parent data window
        )

    def update_data(self, text: str) -> None:
        """Helper method to update displayed text content."""
        self.text_control.value = text