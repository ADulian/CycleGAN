""" A simple tool for image exploration in Notebooks
"""
import matplotlib.pyplot as plt

from ipywidgets import Button, Output, HBox
from IPython.display import display, clear_output

# --------------------------------------------------------------------------------
class ImageExplorer:
    """ Simple tool for viewing images in notebooks

    Implements a n_rows by n_cols plt.figure with up and down buttons to change between rows

    """

    # --------------------------------------------------------------------------------
    def __init__(self, dataset, n_rows=3, n_cols=3, figsize=(12, 12)):
        """ Initialise the Image Explorer

        How to:
            explorer = ImageExplorer(dataset)
            explorer.show()

        Tips: Override any functionality for your own case, I left some comments in other functions
        that might be useful to provide a case specific behaviour

        ---
        Parameters
            dataset: Dataset class, can be anything (torch.Dataset recommended) as long as it implements len()
            n_rows: Number of rows
            n_cols: Number of columns
            figsize: Figure size

        """
        self.dataset = dataset
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.figsize = figsize
        self.total_images = n_rows * n_cols
        self.num_samples = len(dataset)
        self.current_index = 0

        self.output_widget = Output()
        self.button_up = Button(description='▲')
        self.button_down = Button(description='▼')
        self.button_box = HBox([self.button_up, self.button_down], align_items='center')

        self._initialize_callbacks()

    # --------------------------------------------------------------------------------
    def _initialize_callbacks(self):
        """ Initialise callbacks for up and down buttons
        """

        self.button_up.on_click(self._on_up_click)
        self.button_down.on_click(self._on_down_click)

    # --------------------------------------------------------------------------------
    def _on_up_click(self, button):
        """ On up click
        """

        self.current_index -= self.n_cols
        self._update_display()

    # --------------------------------------------------------------------------------
    def _on_down_click(self, button):
        """ On down click
        """

        self.current_index += self.n_cols
        self._update_display()

    # --------------------------------------------------------------------------------
    def _update_display(self):
        """ Update figure
        """

        self.current_index = max(0, min(self.current_index, self.num_samples - self.total_images))
        with self.output_widget:
            clear_output(wait=True)
            self._display_images()

    # --------------------------------------------------------------------------------
    def _display_images(self):
        """ Show grid of images

        Tip: Override this function for your specific behaviour

        """

        fig, axs = plt.subplots(self.n_rows, self.n_cols, figsize=self.figsize)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                idx = self.current_index + i * self.n_cols + j
                if idx < self.num_samples:
                    axs[i, j].imshow(self._get_image(idx=idx))
                    axs[i, j].axis('off')
        plt.show()

    # --------------------------------------------------------------------------------
    def _get_image(self, idx):
        """ Function that retrieves an image at specific idx

        This is just an example of a function that assumes that dataset object imlements __getitem__(idx),
        again, this is a simple PyTorch example

        Tip: Override this function for your specific behaviour. This is called from _display_images and only requires
        to implement functionality that will return some object with which the plt.imshow() is happy with

        ---
        Parameters
            idx: Index of the sample

        ---
        Returns
            Anything that plt.imshow() is happy with

        """

        return self.dataset[idx]

    # --------------------------------------------------------------------------------
    def show(self):
        """ Show the grid
        """

        display(self.output_widget)
        self._update_display()
        display(self.button_box)
