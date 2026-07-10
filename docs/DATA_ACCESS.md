# Dataset Access

This project expects the raw workbook at:

```text
data/raw/Cinnamon_export_sales.xlsx
```

The workbook is not committed to the repository by default because the project constitution treats raw and processed sales data as private. The data includes transaction-level business records and pseudonymous customer fields, so publishing it requires an explicit permission/data-sharing decision.

## How a Visitor Can Run the Project

To run the full pipeline, a visitor needs one of these:

1. The original workbook, provided separately by the project owner, placed at `data/raw/Cinnamon_export_sales.xlsx`.
2. A public/anonymized replacement workbook with the same columns and compatible values.
3. A future synthetic sample dataset generated specifically for public demonstration.

Without a compatible workbook at that path, the data pipeline cannot be reproduced from raw input.

## Required Raw Columns

The pipeline expects these raw columns:

```text
Region
Country
Customer Code
Customer ID
Brand Category
Product Range
Sales Channel
Product Code
Order Date
Invoice Date
Invoice No
Sales USD
Sales Qty
Sales KG
```

## Publishing Decision

Before committing or sharing `Cinnamon_export_sales.xlsx`, confirm that:

- the data owner permits public redistribution;
- transaction-level business data can be made public;
- pseudonymous customer fields are acceptable to publish;
- the assignment or institution allows the dataset to be uploaded to GitHub.

If any of these are uncertain, keep the workbook private and share an anonymized or synthetic version instead.
