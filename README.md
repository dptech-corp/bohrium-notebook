# bohrium-notebook

## 文档结构示例
```
    /-- README.md
    /-- sidebar.json
    /-- molecular_simulation_basic
        /-- 1.ipynb
        /-- 2.ipynb
        /-- 3.ipynb
    /-- molecular_simulation_advanced
        /-- 1.ipynb
        /-- 2.ipynb
        /-- 3.ipynb
```

## sidebar.json 内容示例
```json
[
  {
    "type": "category",
    "id": "molecular_simulation_basic",
    "label": "分子模拟基础",
    "items": [
      {
        "type": "doc",
        "id": "molecular_simulation_basic/1.ipynb",
        "label": "1. 分子模拟简介"
      },
      {
        "type": "doc",
        "id": "molecular_simulation_basic/2.ipynb",
        "label": "2. 分子模拟的基本原理"
      },
      {
        "type": "doc",
        "id": "molecular_simulation_basic/3.ipynb",
        "label": "3. 分子模拟的基本步骤"
      }
    ]
  },
  {
    "type": "category",
    "id": "molecular_simulation_advanced",
    "label": "分子模拟进阶",
    "items": [
      {
        "type": "category",
        "id": "molecular_simulation_advanced/sampler",
        "label": "Sampler",
        "items": [
          {
            "type": "doc",
            "id": "molecular_simulation_advanced/sampler/1.ipynb",
            "label": "1. 分子模拟简介"
          },
          {
            "type": "doc",
            "id": "molecular_simulation_advanced/sampler/2.ipynb",
            "label": "2. 分子模拟的基本原理"
          },
          {
            "type": "doc",
            "id": "molecular_simulation_advanced/sampler/3.ipynb",
            "label": "3. 分子模拟的基本步骤"
          }
        ]
      },
      {
        "type": "category",
        "id": "molecular_simulation_advanced/free_energy_calculation",
        "label": "Free Energy Calculation",
        "items": [
          {
            "type": "doc",
            "id": "molecular_simulation_advanced/free_energy_calculation/1.ipynb",
            "label": "1. 分子模拟简介"
          },
          {
            "type": "doc",
            "id": "molecular_simulation_advanced/free_energy_calculation/2.ipynb",
            "label": "2. 分子模拟的基本原理"
          },
          {
            "type": "doc",
            "id": "molecular_simulation_advanced/free_energy_calculation/3.ipynb",
            "label": "3. 分子模拟的基本步骤"
          }
        ]
      },
      {
        "type": "doc",
        "id": "molecular_simulation_advanced/3.ipynb",
        "label": "3. 分子模拟的基本步骤"
      }
    ]
  }
]

```
