"""Camera pose source reader.

``spatialvid`` reads the shared Parquet + pose7 sidecar contract consumed by
``camera/dataset.py``; ``format`` turns those rows into the canonical
``[F, 12]`` camera tensor.
"""
