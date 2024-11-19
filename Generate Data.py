# Databricks notebook source
# MAGIC %sql
# MAGIC create schema if not exists benmackenzie_catalog.credit_card_fraud_demo

# COMMAND ----------

# MAGIC %sql
# MAGIC use benmackenzie_catalog.credit_card_fraud_demo

# COMMAND ----------

# MAGIC %pip install Faker
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from faker import Faker
import datetime
import random

# Initialize Faker and Spark session
fake = Faker()

# Function to generate synthetic data
def generate_data(num_records):
    data = []
    for customer_id in range(10000, num_records + 10001):
        name = fake.name()
        email = fake.email()
        
        # Generate birth date
        age = random.randint(18, 70)  # Customer age between 18 and 70
        birth_date = datetime.date.today() - datetime.timedelta(days=365 * age)

        # Member since no earlier than 18 years from birth_date
        start_date_for_member_since = birth_date + datetime.timedelta(days=365 * 18)
        member_since = fake.date_between(start_date=start_date_for_member_since, end_date='today')
        
        num_changes = random.randint(1, 3)  # Each customer can have between 1 and 3 address records

        previous_date = member_since
        for change in range(num_changes):
            street = fake.street_address()
            city = fake.city()
            state = fake.state_abbr()
            zip_code = fake.zipcode()
            valid_from = previous_date
            valid_to = fake.date_between(start_date=valid_from, end_date='today') if change < num_changes - 1 else None
            is_current = 1 if change == num_changes - 1 else 0
            previous_date = valid_to

            data.append((customer_id, name, email, street, city, state, zip_code, birth_date, member_since, valid_from, valid_to, is_current))

    return data

# Generate and create DataFrame
num_customers = 100
customers = generate_data(num_customers)
customer_df = spark.createDataFrame(customers, ["customer_id", "name", "email", "street", "city", "state", "zip_code", "birth_date", "member_since", "valid_from", "valid_to", "is_current"])



# COMMAND ----------

customer_df.write.format('delta').mode('overwrite').saveAsTable('customers')


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from faker import Faker
import random

# Initialize Faker
fake = Faker()


# Function to generate merchant data
def generate_merchant_data():
    data = []
    for i in range(100):
        merchant_id = 20000 + i
        name = fake.company()
        street_address = fake.street_address()
        city = fake.city()
        state = fake.state_abbr()
        postal_code = fake.zipcode()
        data.append((merchant_id, name, street_address, city, state, postal_code))
    return data

# Generate merchant data

merchants = generate_merchant_data()
merchant_df = spark.createDataFrame(merchants, ["merchant_id", "name", "street_address", "city", "state", "merchant_zip"])




# COMMAND ----------

merchant_df.write.format('delta').mode('overwrite').option("overwriteSchema", "true").saveAsTable('merchants')

# COMMAND ----------

# MAGIC %md
# MAGIC add merchant id instead of zipcode?
# MAGIC Primary Account Number (PAN): This is the credit card number, which uniquely identifies the cardholder’s account.
# MAGIC
# MAGIC Transaction Amount: The total amount of the transaction, which may include the cost of the purchase and any additional fees (such as taxes or shipping).
# MAGIC
# MAGIC Merchant Identifier: This includes the merchant's name and possibly their merchant ID. This helps Visa identify where the transaction is occurring.
# MAGIC
# MAGIC Service Code: Information about the card's accepted uses and restrictions (e.g., whether it can be used internationally).
# MAGIC
# MAGIC Expiration Date: The card’s expiration date, used to verify that the card is still valid.
# MAGIC
# MAGIC Security Code (CVV, CVC, CID): A 3 or 4-digit code on the card that is not stored in the magnetic stripe, used to verify that the person making the transaction has the card in their possession.
# MAGIC
# MAGIC Date and Time: The exact date and time of the transaction.
# MAGIC
# MAGIC Terminal Identifier: Information about the terminal where the transaction is processed. This includes data like the terminal ID and the location.
# MAGIC
# MAGIC Transaction Type: Specifies the nature of the transaction (e.g., purchase, refund, cash advance).
# MAGIC
# MAGIC Authorization Code: A unique code returned by Visa once they approve the transaction, which is used to finalize the transaction on the merchant’s end.
# MAGIC
# MAGIC Currency Code: The currency in which the transaction is being made.
# MAGIC
# MAGIC POS Entry Mode: Point-of-sale (POS) entry mode indicates how the card information was captured (e.g., swiped, inserted, tapped, manually entered).

# COMMAND ----------


from faker import Faker
import datetime
import random

from pyspark.sql.types import ArrayType, StructType, StructField, IntegerType, DateType, BooleanType, FloatType

from pyspark.sql.functions import udf, explode, col


# Initialize Faker and Spark session
fake = Faker()

# Function to calculate number of years
def calculate_years(start_date):
    end_date = datetime.date.today()
    return (end_date - start_date).days // 365

transaction_id = 1000000

# Function to generate transactions for a single customer
def generate_transactions(member_since):
    global transaction_id
    num_years = calculate_years(member_since)
    num_transactions = 50 * num_years  # approximately 50 transactions per year
    security_code = random.randint(101,999)
    transactions = []
    for _ in range(num_transactions):

        date = fake.date_between(start_date=member_since, end_date=datetime.date.today())
        amount = round(random.uniform(5.0, 500.0), 2)  # Transaction amount between $5 and $500
        card_present = random.choice([True, False])  # Card present transaction or not
        pos_entry_mode = random.choice([1,2,3,4])
        merchant_id = random.randint(20000,20100)
       
        fraudulent = random.choices([True, False], weights=[1, 99], k=1)[0]  # 1% chance of being fraudulent
        transactions.append((transaction_id, date, amount, card_present, pos_entry_mode, merchant_id, security_code, fraudulent))
        transaction_id += random.randint(0,100)
    return transactions

transaction_schema = ArrayType(StructType([
    StructField("transaction_id", IntegerType(), False),
    StructField("transaction_date", DateType(), False),
    StructField("amount", FloatType(), False),
    StructField("card_present", BooleanType(), False),
    StructField("pos_entry_mode", IntegerType(), False),
    StructField("merchant_id", IntegerType(), False),
    StructField("security_code", IntegerType(), False),
    StructField("fraudulent", BooleanType(), False)
]))


generate_transactions_udf = udf(generate_transactions, transaction_schema)

customer_df = spark.sql('select * from customers')

df = customer_df.withColumn("transactions", generate_transactions_udf("member_since"))

df_exploded = df.withColumn("transaction", explode("transactions"))

# Select and rename columns as required
transaction_df = df_exploded.select(
    col("customer_id").alias("primary_account_number"),
    col("transaction.transaction_id"),
    col("transaction.transaction_date"),
    col("transaction.amount"),
    col("transaction.card_present"),
    col("transaction.pos_entry_mode"),
    col("transaction.merchant_id"),
    col("transaction.security_code"),
    col("transaction.fraudulent")
)







# COMMAND ----------

transaction_df.write.format('delta').mode('overwrite').option("overwriteSchema", "true").saveAsTable('transactions')

# COMMAND ----------

# MAGIC %md
# MAGIC just create random 7 day, 14 day aggregates

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, sequence, expr, col, to_date, lit, coalesce, round as _round, sum as _sum, count, when, max
from datetime import date
from pyspark.sql.window import Window

# Assuming 'transaction_df' and 'customer_df' are your DataFrame loaded appropriately

# Extend 'member_since' date to current date for each customer
current_date = date.today()
all_dates_df = customer_df.select("customer_id", "member_since").withColumn("current_date", lit(current_date))
all_dates_df = all_dates_df.withColumn("date_range", explode(sequence(to_date("member_since"), to_date("current_date"))))

# Ensure transaction_df has 'transaction_date' as a date type and handle null amounts
transaction_df = transaction_df.withColumn("transaction_date", to_date(col("transaction_date")))
transaction_df = transaction_df.withColumn("amount", coalesce(_round("amount", 2), lit(0)))

# Join all_dates_df with transaction_df
full_transaction_df = all_dates_df.join(transaction_df, (all_dates_df.customer_id == transaction_df.primary_account_number) & (all_dates_df.date_range == transaction_df.transaction_date), "left_outer")
full_transaction_df = full_transaction_df.select(all_dates_df.customer_id, all_dates_df.date_range.alias("date"), coalesce(transaction_df.amount, lit(0)).alias("amount"))

full_transaction_df = full_transaction_df.withColumn("transaction_occurred", when(col("amount") > 0, 1).otherwise(0))


# Aggregate daily transactions by customer and date to ensure one entry per day per customer
daily_transaction_df = full_transaction_df.groupBy("customer_id", "date").agg(_sum("amount").alias("daily_total"),max("transaction_occurred").alias("transaction_occurred"))

# Define the window specification for rolling windows
windowSpec_14 = Window.partitionBy("customer_id").orderBy("date").rowsBetween(-13, 0)
windowSpec_7 = Window.partitionBy("customer_id").orderBy("date").rowsBetween(-6, 0)

# Calculate the rolling sums and counts over the specified windows
daily_transaction_df = daily_transaction_df.withColumn("14_day_total", _sum("daily_total").over(windowSpec_14))
daily_transaction_df = daily_transaction_df.withColumn("7_day_total", _sum("daily_total").over(windowSpec_7))
daily_transaction_df = daily_transaction_df.withColumn("7_day_transaction_count", _sum("transaction_occurred").over(windowSpec_7))

# Display or further process the results
daily_transaction_df.select("customer_id", "date", "14_day_total", "7_day_total", "7_day_transaction_count").show(truncate=False)


daily_transaction_df = daily_transaction_df.select("customer_id", "date", _round(col("14_day_total"),2).alias("14_day_total"),
                                                   _round(col("7_day_total"),2).alias("7_day_total"), "7_day_transaction_count")


# COMMAND ----------

daily_transaction_df.write.format('delta').mode('overwrite').option("overwriteSchema", "true").saveAsTable('transaction_aggregates')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from transaction_aggregates

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE customers ALTER COLUMN customer_id SET NOT NULL;
# MAGIC ALTER TABLE customers ALTER COLUMN valid_from SET NOT NULL;
# MAGIC ALTER TABLE customers ADD CONSTRAINT pk_customers PRIMARY KEY(customer_id, valid_from TIMESERIES);
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE merchants ALTER COLUMN merchant_id SET NOT NULL;
# MAGIC ALTER TABLE merchants ADD CONSTRAINT pk_merchants PRIMARY KEY(merchant_id);

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE transaction_aggregates ALTER COLUMN customer_id SET NOT NULL;
# MAGIC ALTER TABLE transaction_aggregates ALTER COLUMN date SET NOT NULL;
# MAGIC ALTER TABLE transaction_aggregates ADD CONSTRAINT transaction_aggregates_pk PRIMARY KEY(customer_id, date TIMESERIES);

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE transactions ALTER COLUMN transaction_id SET NOT NULL;
# MAGIC ALTER TABLE transactions ADD CONSTRAINT pk_transactions PRIMARY KEY(transaction_id);
# MAGIC
# MAGIC