#!/usr/bin/env python
# coding: utf-8

# # market basket analysis
# 
# 

# In[1]:


##import packages for data visualization and do #apriori algorithm


# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules 
import warnings
warnings.filterwarnings("ignore", message="Specific warning message you want to ignore")


import seaborn as sns


# In[36]:


##loading the dataset


# In[37]:


data=pd.read_excel("Assignment-1_Data.xlsx")
data


# In[38]:


data['BillNo'] =  data['BillNo'].astype('str')
data= data[~data['BillNo'].str.contains('C')]


# In[39]:


##data preprocessing


# In[40]:


###identifying any duplicates entry in the dataset


# In[41]:


df = pd.DataFrame(data)
df= df.drop_duplicates()


# In[42]:


df


# In[43]:


if df.duplicated().any().any():
    print("Duplicates found in the dataset")
else:
    print("No duplicates found in the dataset")


# In[44]:


##identifying any missing data found in the dataset


# In[45]:


missing_values = data.isnull().sum()


# In[46]:


missing_values


# In[47]:


##filling the missing data using mode function in the respected columns


# In[48]:


data_cleaned = df.dropna(subset=["CustomerID"])
mode_itemname = data_cleaned['Itemname'].mode()[0]
data_cleaned = data_cleaned.copy() 
data_cleaned['Itemname'].fillna(mode_itemname, inplace=True)
data_cleaned.head()


# In[49]:


data_cleaned.isnull().sum()


# In[50]:


df=data_cleaned


# In[51]:


## top 25 frequently bought items by customer


# In[52]:


plt.rcParams['figure.figsize']=25,7
sns.countplot(data=df,x=df['Itemname'],order=df['Itemname'].value_counts().head(25).index)
plt.xticks(rotation=75)
plt.xlabel('product')
plt.title('top 25 frequently bought products')
plt.show()


# In[53]:


# calculating the sales trend based on the year


# In[54]:


df[df["Date"].dt.year==2010].groupby(df["Date"].dt.month)["Price"].sum().plot()
df[df["Date"].dt.year==2011].groupby(df["Date"].dt.month)["Price"].sum().plot()
plt.legend(['2010','2011'])
plt.title("income over time")
plt.ylabel('Total income(million)')
plt.xlabel("Date(month)")


# In[55]:


##assign the original dataframe to df2


# In[56]:


df2=df
#filter rows based on item occurences
item_counts=df2['Itemname'].value_counts(ascending=False)
filtered_items=item_counts.loc[item_counts >1].reset_index()['index']
df2=df2[df2['Itemname'].isin(filtered_items)]
#filter rows based on bill number occurences

bill_counts=df2['BillNo'].value_counts(ascending=False)
filtered_bills=bill_counts.loc[bill_counts > 1].reset_index()['index']
df2=df2[df2['BillNo'].isin(filtered_bills)]


# In[57]:


df2


# # generate association rules
# 

# In[58]:


basket = (df2[df2['Country'] == 'Germany' ].groupby(['BillNo','Itemname'])['Quantity'].sum().unstack().fillna(0))


# In[59]:


basket


# In[60]:


df2 = df2[df2['Country'] == 'Germany']
df2


# In[61]:


# Create a pivot table using the filtered DataFrame
pivot_table = pd.pivot_table(df2[['BillNo','Itemname']], index='BillNo', columns='Itemname', aggfunc=lambda x: True, fill_value=False)


# In[62]:


pivot_table


# In[30]:



# Generate frequent itemsets with minimum support of 0.1 (10%)
frequent_itemsets = apriori(pivot_table, min_support=0.05,use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, "confidence", min_threshold = 0.5)

# Print frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Print association rules
print("\nAssociation Rules:")
rules


# In[31]:


rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 

rules


# In[32]:


rules.sort_values(by='support', ascending=False)


# In[33]:


# Sort rules by support in descending order
sorted_rules = rules.sort_values(by='support', ascending=False)

# Calculate cumulative support
cumulative_support = np.cumsum(sorted_rules['support'] / np.sum(sorted_rules['support']) * 100)

# Bar plot for Support
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.bar(range(len(sorted_rules)), sorted_rules['support'], align='center' ,color='#FFF219')


plt.xticks(range(len(sorted_rules)), ['' for _ in range(len(sorted_rules))])  # Remove x-axis labels
ax1.set_xlabel('Association Rule')
ax1.set_ylabel('Support')
ax1.set_title('Support of Association Rules')

# CDF plot for cumulative support
ax2 = ax1.twinx()
ax2.plot(range(len(sorted_rules)), cumulative_support, color='#000000', linestyle='--')
ax2.set_ylabel('Cumulative Support (%)', c='#000000')

plt.tight_layout()
plt.show()

# Scatter plot for Confidence vs. Support
plt.figure(figsize=(8, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.4)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Confidence vs. Support of Association Rules')
plt.tight_layout()
plt.show()


# In[34]:


# Filter association rules for cross-selling opportunities
cross_selling_rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)]

# Sort rules based on confidence and support
cross_selling_rules = cross_selling_rules.sort_values(by=['confidence', 'support'], ascending=False)

# Select top cross-selling recommendations
top_cross_selling = cross_selling_rules.head(5)

# Filter association rules for upselling opportunities
upselling_rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) > 1)]

# Sort rules based on confidence and support
upselling_rules = upselling_rules.sort_values(by=['confidence', 'support'], ascending=False)

# Select top upselling recommendations
top_upselling = upselling_rules.head(5)

# Display cross-selling recommendations
print("Cross-Selling Recommendations:")
for idx, row in top_cross_selling.iterrows():
    antecedent = list(row['antecedents'])[0]
    consequent = list(row['consequents'])[0]
    print(f"Customers who bought '{antecedent}' also bought '{consequent}'.")

# Display upselling recommendations
print("\nUpselling Recommendations:")
for idx, row in top_upselling.iterrows():
    antecedent = list(row['antecedents'])[0]
    consequents = list(row['consequents'])
    print(f"For customers who bought '{antecedent}', recommend the following upgrades: {', '.join(consequents)}.")

