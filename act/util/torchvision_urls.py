#===- act/util/torchvision_urls.py - Torchvision URL Configuration ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===------------------------------------------------------------------------===#
#
# Purpose:
#   Centralised torchvision compatibility patches.
#   Import and call patch_mnist_mirrors() before any code that downloads MNIST,
#   regardless of whether it runs in-process or via subprocess.
#
#===------------------------------------------------------------------------===#

import torchvision.datasets as _tvd

# Reliable MNIST mirrors that replace the defunct yann.lecun.com host.
_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",   # PyTorch / Meta S3
    "https://storage.googleapis.com/cvdf-datasets/mnist/",  # Google CVDF
]


def configure_mirror_urls() -> None:
    """Configure torchvision.datasets.MNIST.mirrors to use reliable download URLs.

    The original mirror (yann.lecun.com) is no longer reliably reachable.
    Call this function once, early in any entry point that may trigger an MNIST
    download (either directly via torchvision or transitively through abcrown /
    ERAN data loaders).

    This is idempotent – calling it multiple times has no side effects.
    """
    _tvd.MNIST.mirrors = list(_MNIST_MIRRORS)
