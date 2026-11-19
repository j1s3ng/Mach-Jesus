# External artifact catalog

[downloads.json](downloads.json) preserves expected detector, driver, reference,
training-bundle and original-delivery identities. Its `source` paths and
`exists` fields describe verified artifacts in the original development
project; they do not assert that those files are in this checkout.

Model/data download URLs are not hosted here. The source repository includes
compact [CPU reproduction inputs](../reproduction/README.md) and
[measured examples](../examples/measured), but full neural inference, training
and physical races require separately obtained hash-matched artifacts.

Follow [setup and restoration](../docs/setup.md) for endpoint configuration and
the flat companion manifest. [Experiment manifests](../experiments/README.md)
pin any additional line/road/runtime dependencies used by an engineering demo.
External models and assets retain their own publication and license terms.
