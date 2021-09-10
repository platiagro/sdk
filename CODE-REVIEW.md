## Dicas para revisão de código

### Commits
- Título (1a linha do commit): apresentar resumo do que foi alterado/adicionado/removido.
ex: adiciona action que salva parametros no backend; exibe rótulo no componente de selecao de dataset;
- Descrição (outras linhas): dar mais detalhes de cada alteração:
- motivos das alterações
    ex: havia um bug que causava...; nova funcionalidade que faz isso...; código foi movido para...;
- bibliotecas adicionadas e versões (requirements.txt)
    ex: atualiza para minio 6.0.0;
- testes unitários criados/alterados
    ex: adiciona testes para o método stat_dataset;
- alterações do `docs/source/platiagro.rst`
    ex: adiciona documentação para save_metrics
- Mensagens auto-explicativas! Quem revisa o código deve entender o que foi feito (e porque foi feito) **sem perguntar para quem fez o commit**.
- Não devem ter conflitos. Solicitar que sejam resolvidas as ocorrências de "This branch has conflicts that must be resolved".

### SonarCloud Quality Gate
- Coverage > 80.0%, e sempre que possível = 100%
- 0 Bugs, 0 Code Smells, 0 Vulnerabilities

### Build Github actions COM SUCESSO

### Python
- Usar Python 3.8
- Remover `print`.
- Não deixar código-fonte comentado.
- f-string `f'text-{variable}'` é melhor que `'text-{}'.format(variable)` e `'text-' + variable`
- Métodos que são chamados de outros arquivos `.py` **DEVEM TER Docstring**.
- Usar Google Style Python Docstring: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
- Verificar se código-fonte e `docs/source/platiagro.rst` estão "compatíveis".
- TODOS OS MÉTODOS EM `platiagro/__init__.py` **DEVEM ESTAR ATUALIZADOS NOS DOCS**.
- Usar sempre import absoluto.
ex: `from platiagro.featuretypes import CATEGORICAL` (BOM), `from .featuretypes import CATEGORICAL` (RUIM)

## Testes Unitários
- Todo teste deve ter um docstring descrevendo o que é testado:
```python
def test_something_exception(self):
    """ 
    Should raise an exception when ...
    """
```
- Cada função deve preferencialmente testar 1 única chamada:
```python
 def test_something_success(self): 
    """ 
    Should return ok.
    """
    result = something()
    assert result == "ok"

def test_something_exception(self):
    """ 
    Should raise an exception.
    """
    with self.assertRaises(Exception):
        something()
```
- Utilize [tests/util.py](./tests/util.py) para códigos de teste reutilizáveis.
```python
import tests.util as util

@mock.patch.object(
    MINIO_CLIENT,
    "get_object",
    side_effect=util.NO_SUCH_KEY_ERROR,
)
```
- Quando criar um mock de uma função, use o [`assert_any_call`](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.assert_any_call) ou [`assert_called_with`](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.assert_called_with) para testar se a função recebeu os parâmetros adequados.<br>
Caso algum dos parâmetros tenha valor dinâmico, ou possa ser ignorado, utilize [`mock.ANY`](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.ANY).
```python
@mock.patch.object( 
    MINIO_CLIENT, 
    "put_object", 
    side_effect=util.put_object_side_effect, 
) 
def test_update_dataset_metadata( 
    self, mock_put_object
):
    ...
    mock_put_object.assert_any_call( 
        bucket_name=BUCKET_NAME, 
        object_name=f"datasets/{dataset_name}/{dataset_name}.metadata", 
        data=mock.ANY, 
        length=mock.ANY, 
    ) 
```
