#!/bin/bash
# Fix convert_expert_data.py on remote machine

# Backup original
cp convert_expert_data.py convert_expert_data.py.backup

# Apply fix
sed -i 's/env = RLFactory.make(/cpu_env_name = args.env_name.replace("Mjx", "")\n    env = RLFactory.make(/' convert_expert_data.py
sed -i 's/args.env_name,/cpu_env_name,/' convert_expert_data.py

echo "Fixed convert_expert_data.py"
echo "Check the diff:"
diff -u convert_expert_data.py.backup convert_expert_data.py
