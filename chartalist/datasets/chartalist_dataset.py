import os
import time
class ChartaListDataset:
    """
    Shared dataset class for all chartalist datasets.
    """
    DEFAULT_SPLITS = {'train': 0, 'val': 1, 'test': 2}
    DEFAULT_SPLIT_NAMES = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
    DEFAULT_SOURCE_DOMAIN_SPLITS = [0]

    def __init__(self, root_dir, download, split_scheme):
        self.check_init()


    def __getitem__(self, idx):
        # Any transformations are handled by the ChartaListSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        return x

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        raise NotImplementedError



    def check_init(self):
        """
        Convenience function to check that the ChartaListDataset is properly configured.
        """
        required_attrs = ['_dataset_name', '_data_dir']
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'ChartaListDataset is missing {attr_name}.'

        # Check that data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

    @property
    def latest_version(cls):
        def is_later(u, v):
            """Returns true if u is a later version than v."""
            u_major, u_minor = tuple(map(int, u.split('.')))
            v_major, v_minor = tuple(map(int, v.split('.')))
            if (u_major > v_major) or (
                    (u_major == v_major) and (u_minor > v_minor)):
                return True
            else:
                return False

        latest_version = '0.0'
        for key in cls.versions_dict.keys():
            if is_later(key, latest_version):
                latest_version = key
        return latest_version

    @property
    def dataset_name(self):
        """
        A string that identifies the dataset'.
        """
        return self._dataset_name

    @property
    def version(self):
        """
        A string that identifies the dataset version, e.g., '1.0'.
        """
        if self._version is None:
            return self.latest_version
        else:
            return self._version

    @property
    def versions_dict(self):
        """
        A dictionary where each key is a version string (e.g., '1.0')
        and each value is a dictionary containing the 'download_url' and
        'compressed_size' keys.

        'download_url' is the URL for downloading the dataset archive.
        If None, the dataset cannot be downloaded automatically
        (e.g., because it first requires accepting a usage agreement).

        'compressed_size' is the approximate size of the compressed dataset in bytes.
        """
        return self._versions_dict

    @property
    def data_dir(self):
        """
        The full path to the folder in which the dataset is stored.
        """
        return self._data_dir

    @property
    def data_frame(self):
        """
        The full path to the folder in which the dataset is stored.
        """
        return self._data_frame

    @property
    def collate(self):
        """
        Torch function to collate items in a batch.
        By default returns None -> uses default torch collate.
        """
        return getattr(self, '_collate', None)






    @property
    def source_domain_splits(self):
        """
        List of split IDs that are from the source domain.
        """
        return getattr(self, '_source_domain_splits', ChartaListDataset.DEFAULT_SOURCE_DOMAIN_SPLITS)



    @property
    def n_classes(self):
        """
        Number of classes for single-task classification datasets.
        Used for logging and to configure models to produce appropriately-sized output.
        None by default.
        Leave as None if not applicable (e.g., regression or multi-task classification).
        """
        return getattr(self, '_n_classes', None)

    @property
    def is_classification(self):
        """
        Boolean. True if the task is classification, and false otherwise.
        """
        return getattr(self, '_is_classification', (self.n_classes is not None))

    @property
    def is_detection(self):
        """
        Boolean. True if the task is detection, and false otherwise.
        """
        return getattr(self, '_is_detection', False)

    @property
    def original_resolution(self):
        """
        Original image resolution for image datasets.
        """
        return getattr(self, '_original_resolution', None)

    def initialize_data_dir(self, root_dir, download, file_name):
        """
        Helper function for downloading/updating the dataset if required.
        Note that we only do a version check for datasets where the download_url is set.
        Currently, this includes all datasets except Yelp.
        Datasets for which we don't control the download, like Yelp,
        might not handle versions similarly.
        """
        self.check_version()

        os.makedirs(root_dir, exist_ok=True)
        data_dir = os.path.join(root_dir, f'{self.dataset_name}_{self.version}')
        version_file = os.path.join(data_dir, f'RELEASE_{self.version}.txt')

        # If the dataset exists at root_dir, then don't download.
        if not self.dataset_exists_locally(data_dir, version_file, file_name):
            self.download_dataset(data_dir, download)
        return data_dir

    def dataset_exists_locally(self, data_dir, version_file, file_name):
        download_url = self.versions_dict[self.version]['download_url']
        # There are two ways to download a dataset:
        # 1. Automatically through the chartalist package
        # 2. From a third party (e.g. OGB-MolPCBA is downloaded through the OGB package)
        # Datasets downloaded from a third party need not have a download_url and RELEASE text file.
        return (
                os.path.exists(data_dir) and (
                os.path.exists(version_file) or
                (len(os.listdir(data_dir)) > 0 and download_url is None)
                or os.path.exists(data_dir + "\\" + file_name)
        )
        )

    def download_dataset(self, data_dir, download_flag):
        version_dict = self.versions_dict[self.version]
        download_url = version_dict['download_url']
        compressed_size = version_dict['compressed_size']

        # Check that download_url exists.
        if download_url is None:
            raise ValueError(f'{self.dataset_name} cannot be automatically downloaded. Please download it manually.')

        # Check that the download_flag is set to true.
        if not download_flag:
            raise FileNotFoundError(
                f'The {self.dataset_name} dataset could not be found in {data_dir}. Initialize the dataset with '
                f'download=True to download the dataset. If you are using the example script, run with --download. '
                f'This might take some time for large datasets.'
            )

        from chartalist.datasets.download_utils import download_and_extract_archive
        print(f'Downloading dataset to {data_dir}...')
        print(f'You can also download the dataset manually at https://chartalist.org/.')

        try:
            start_time = time.time()
            download_and_extract_archive(
                url=download_url,
                download_root=data_dir,
                remove_finished=True,
                size=compressed_size)
            download_time_in_minutes = (time.time() - start_time) / 60
            print(f"\nIt took {round(download_time_in_minutes, 2)} minutes to download and uncompress the dataset.\n")
        except Exception as e:
            print(
                f"\n{os.path.join(data_dir, 'archive.tar.gz')} may be corrupted. Please try deleting it and rerunning this command.\n")
            print(f"Exception: ", e)

    def check_version(self):
        # Check that the version is valid.
        if self.version not in self.versions_dict:
            raise ValueError(f'Version {self.version} not supported. Must be in {self.versions_dict.keys()}.')




