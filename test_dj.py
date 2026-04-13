from lightphe import LightPHE
import os

print("=== Testing LightPHE Damgård-Jurik Key Save/Load API ===")

# 1. Generate keys for Damgård-Jurik
print("\n[1] Generating DJ keys (default s=2)...")
cs1 = LightPHE(algorithm_name="Damgard-Jurik", key_size=1024)
s_param = cs1.cs.keys.get("public_key", {}).get("s")
print(f"-> Key generation complete. 's' parameter is: {s_param}")

# 2. Encrypt some data
m1, m2 = 15, 25
c1 = cs1.encrypt(m1)
c2 = cs1.encrypt(m2)
print(f"-> Encrypted {m1} and {m2}")

# 3. Export keys to file
key_file = "dj_keys.json"
print(f"\n[2] Exporting keys to {key_file}...")
# LightPHE warns about exporting private keys if public is not True.
# But we need to save the private key for decryption.
cs1.export_keys(target_file=key_file, public=False)
print(f"-> Keys successfully exported.")

# 4. Load keys into a new instance
print("\n[3] Restoring keys into a new LightPHE instance...")
cs2 = LightPHE(algorithm_name="Damgard-Jurik")
cs2.restore_keys(target_file=key_file)
print(f"-> Keys successfully restored.")

# 5. Verify the keys match and homomorphic operations work
c_sum = c1 + c2
decrypted_sum = cs2.decrypt(c_sum)
print(f"\n[4] Verifying with homomorphic addition...")
print(f"-> Decrypted sum of c1 and c2 using restored keys: {decrypted_sum}")
assert decrypted_sum == m1 + m2, "Decryption failed or incorrect!"
print("-> Success! The save/load API works perfectly.")

# Cleanup
if os.path.exists(key_file):
    os.remove(key_file)
