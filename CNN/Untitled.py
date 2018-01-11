
# coding: utf-8

# In[3]:


# test editing yml file
import yaml


# In[4]:


def set_state(state):
    with open('init_args_test.yml') as f:
        doc = yaml.load(f)

    doc['target_size'] = state

    with open('init_args_test.yml', 'w') as f:
        yaml.dump(doc, f)


# In[5]:


set_state('200')


# In[8]:


with open("init_args.yml", 'r') as stream:
            try:
                init_args = yaml.load(stream)
                print("init_args_type",type(init_args))
                print(init_args)
            except yaml.YAMLError as exc:
                print(exc)

