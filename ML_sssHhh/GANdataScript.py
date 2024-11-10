import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

class IndianLifestyleGAN:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.n_features = 40  # Increased features to accommodate more device types
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        
    def build_generator(self):
        model = keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(self.n_features, activation='tanh')
        ])
        return model
        
    def build_discriminator(self):
        model = keras.Sequential([
            layers.Input(shape=(self.n_features,)),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    def build_gan(self):
        self.discriminator.trainable = False
        model = keras.Sequential([self.generator, self.discriminator])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

class IndianFamilyLifestyleGenerator:
    def __init__(self):
        # Define Indian family profiles
        self.profiles = [
            {
                "id": "FAM001_FATHER",
                "name": "Rajesh Kumar",
                "age": 42,
                "role": "Father",
                "occupation": "IT Professional",
                "work_schedule": "9-6",
                "lifestyle": {
                    "fitness_level": "moderate",
                    "diet_type": "vegetarian",
                    "stress_level": "high",
                    "sleep_quality": "moderate",
                    "social_activity": "moderate"
                }
            },
            {
                "id": "FAM001_MOTHER",
                "name": "Priya Kumar",
                "age": 38,
                "role": "Mother",
                "occupation": "School Teacher",
                "work_schedule": "8-3",
                "lifestyle": {
                    "fitness_level": "moderate",
                    "diet_type": "vegetarian",
                    "stress_level": "moderate",
                    "sleep_quality": "moderate",
                    "social_activity": "high"
                }
            },
            {
                "id": "FAM001_SON",
                "name": "Arjun Kumar",
                "age": 16,
                "role": "Son",
                "occupation": "Student",
                "work_schedule": "school_hours",
                "lifestyle": {
                    "fitness_level": "high",
                    "diet_type": "vegetarian",
                    "stress_level": "moderate",
                    "sleep_quality": "good",
                    "social_activity": "very_high"
                }
            },
            {
                "id": "FAM001_DAUGHTER",
                "name": "Ananya Kumar",
                "age": 8,
                "role": "Daughter",
                "occupation": "Student",
                "work_schedule": "school_hours",
                "lifestyle": {
                    "fitness_level": "high",
                    "diet_type": "vegetarian",
                    "stress_level": "low",
                    "sleep_quality": "good",
                    "social_activity": "high"
                }
            },
            {
                "id": "FAM001_GRANDMOTHER",
                "name": "Kamala Devi",
                "age": 68,
                "role": "Grandmother",
                "occupation": "Retired",
                "work_schedule": "home",
                "lifestyle": {
                    "fitness_level": "low",
                    "diet_type": "vegetarian",
                    "stress_level": "low",
                    "sleep_quality": "moderate",
                    "social_activity": "moderate"
                }
            },
            {
                "id": "FAM001_GRANDFATHER",
                "name": "Mohan Kumar",
                "age": 72,
                "role": "Grandfather",
                "occupation": "Retired",
                "work_schedule": "home",
                "lifestyle": {
                    "fitness_level": "low",
                    "diet_type": "vegetarian",
                    "stress_level": "low",
                    "sleep_quality": "moderate",
                    "social_activity": "moderate"
                }
            }
        ]
        
        # Initialize GAN
        self.gan = IndianLifestyleGAN()
        
        # Define Indian household specific device and appliance usage patterns
        self.device_patterns = {
            "kitchen_appliances": {
                "mixer_grinder": (0, 45),      # minutes per use
                "pressure_cooker": (0, 30),    # minutes per use
                "microwave": (0, 15),          # minutes per use
                "refrigerator": (24, 24),      # hours (constant)
                "water_purifier": (0, 24),     # hours
                "induction_cooktop": (0, 60)   # minutes per use
            },
            "entertainment_devices": {
                "tv_usage": (0, 300),          # minutes
                "smartphone": (0, 360),        # minutes
                "laptop": (0, 480),            # minutes
                "tablet": (0, 180)             # minutes
            },
            "comfort_appliances": {
                "air_conditioner": (0, 12),    # hours (seasonal)
                "air_cooler": (0, 12),         # hours (seasonal)
                "ceiling_fan": (0, 24),        # hours
                "water_heater": (0, 30)        # minutes
            },
            "other_appliances": {
                "washing_machine": (0, 60),    # minutes per use
                "iron": (0, 30),               # minutes
                "vacuum_cleaner": (0, 30),     # minutes
                "water_pump": (0, 60)          # minutes
            }
        }
        
        # Define seasonal patterns (Indian context)
        self.seasons = {
            "summer": {
                "air_conditioner_multiplier": 1.5,
                "fan_multiplier": 1.8,
                "water_usage_multiplier": 1.4
            },
            "monsoon": {
                "air_conditioner_multiplier": 0.6,
                "fan_multiplier": 0.7,
                "water_usage_multiplier": 0.8
            },
            "winter": {
                "air_conditioner_multiplier": 0.1,
                "fan_multiplier": 0.3,
                "water_usage_multiplier": 0.7
            }
        }
        
        # Define additional features
        self.additional_features = {
            "eating_habits": {
                "vegetarian": 0.8,
                "non_vegetarian": 1.2,
                "vegan": 0.7
            },
            "vehicle_and_travel": {
                "vehicle_monthly_distance_km": (0, 2000),
                "frequency_of_traveling_by_air": {
                    "very_frequently": 1.5,
                    "frequently": 1.2,
                    "rarely": 0.8,
                    "never": 0.5
                },
                "vehicle_type": {
                    "electric": 0.5,
                    "hybrid": 0.7,
                    "petrol": 1.2,
                    "diesel": 1.5,
                    "lpg": 1.0
                }
            },
            "day_to_day_activities": {
                "new_clothes_monthly": (0, 10),
                "waste_bag_weekly_count": (0, 7),
                "waste_bag_size": {
                    "small": 0.5,
                    "medium": 1.0,
                    "large": 1.5,
                    "extra_large": 2.0
                }
            }
        }
        
        # Define an extensive list of Indian cuisines and their carbon footprint impact (kg CO2 per serving)
        self.food_items = {
            "vegetarian": {
                "rice": 0.1,
                "dal": 0.05,
                "vegetables": 0.03,
                "fruits": 0.02,
                "milk": 0.2,
                "paneer": 0.3,
                "roti": 0.04,
                "sambar": 0.06,
                "idli": 0.05,
                "dosa": 0.07,
                "poha": 0.04,
                "upma": 0.05,
                "curd": 0.1
            },
            "non_vegetarian": {
                "chicken": 2.5,
                "fish": 1.5,
                "eggs": 0.6,
                "mutton": 3.0,
                "prawns": 2.0,
                "rice": 0.1,
                "vegetables": 0.03,
                "roti": 0.04,
                "biryani": 1.0,
                "butter_chicken": 2.8,
                "fish_curry": 1.7
            },
            "vegan": {
                "rice": 0.1,
                "dal": 0.05,
                "vegetables": 0.03,
                "fruits": 0.02,
                "soy_milk": 0.1,
                "tofu": 0.2,
                "roti": 0.04,
                "sambar": 0.06,
                "idli": 0.05,
                "dosa": 0.07,
                "poha": 0.04,
                "upma": 0.05
            }
        }
        
        # Ensure output directory exists
        self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def get_season(self, date):
        month = date.month
        if 3 <= month <= 6:
            return "summer"
        elif 7 <= month <= 9:
            return "monsoon"
        else:
            return "winter"

    def generate_hourly_data(self, profile, date_hour):
        """Generate synthetic hourly data for an Indian family member"""
        hour = date_hour.hour
        season = self.get_season(date_hour)
        is_weekend = date_hour.weekday() >= 5
        
        # Generate base data using GAN
        noise = np.random.normal(0, 1, (1, self.gan.latent_dim))
        gan_data = self.gan.generator.predict(noise, verbose=0)[0]
        
        # Initialize activities dictionary
        activities = {}
        
        # Kitchen appliance usage with realistic reasoning
        activities.update({
            "mixer_grinder_usage": self._scale_value(gan_data[0], *self.device_patterns["kitchen_appliances"]["mixer_grinder"]) 
                if 6 <= hour <= 20 else 0,
            "pressure_cooker_usage": self._scale_value(gan_data[1], *self.device_patterns["kitchen_appliances"]["pressure_cooker"]) 
                if (6 <= hour <= 9) or (17 <= hour <= 19) else 0,
            "microwave_usage": self._scale_value(gan_data[2], *self.device_patterns["kitchen_appliances"]["microwave"]) 
                if (7 <= hour <= 9) or (12 <= hour <= 14) or (18 <= hour <= 20) else 0,
            "refrigerator_usage": 1.0,  # Constant operation
            "water_purifier_usage": self._scale_value(gan_data[3], *self.device_patterns["kitchen_appliances"]["water_purifier"]) 
                if 6 <= hour <= 22 else 0
        })
        
        # Entertainment device usage (adjusted for role and time of day)
        activities.update({
            "tv_usage": self._adjust_entertainment_usage(profile, hour, gan_data[4], "tv_usage"),
            "smartphone_usage": self._adjust_entertainment_usage(profile, hour, gan_data[5], "smartphone"),
            "laptop_usage": self._adjust_entertainment_usage(profile, hour, gan_data[6], "laptop"),
            "tablet_usage": self._adjust_entertainment_usage(profile, hour, gan_data[7], "tablet")
        })
        
        # Comfort appliance usage (with seasonal adjustments)
        season_multiplier = self.seasons[season]
        activities.update({
            "air_conditioner_usage": self._scale_value(gan_data[8], *self.device_patterns["comfort_appliances"]["air_conditioner"]) 
                * season_multiplier["air_conditioner_multiplier"] if season == "summer" else 0,
            "fan_usage": self._scale_value(gan_data[9], *self.device_patterns["comfort_appliances"]["ceiling_fan"]) 
                * season_multiplier["fan_multiplier"],
            "water_heater_usage": self._scale_value(gan_data[10], *self.device_patterns["comfort_appliances"]["water_heater"]) 
                * (1.5 if season == "winter" else 0.2)
        })
        
        # Add additional features
        activities.update({
            "eating_habits": self.additional_features["eating_habits"][profile["lifestyle"]["diet_type"]],
            "vehicle_monthly_distance_km": self._scale_value(gan_data[11], *self.additional_features["vehicle_and_travel"]["vehicle_monthly_distance_km"]),
            "frequency_of_traveling_by_air": self._get_travel_frequency(profile),
            "vehicle_type": self._get_vehicle_type(profile),
            "new_clothes_monthly": self._scale_value(gan_data[12], *self.additional_features["day_to_day_activities"]["new_clothes_monthly"]),
            "waste_bag_weekly_count": self._scale_value(gan_data[13], *self.additional_features["day_to_day_activities"]["waste_bag_weekly_count"]),
            "waste_bag_size": self._get_waste_bag_size(profile)
        })
        
        # Add food items and their carbon footprint
        food_type = profile["lifestyle"]["diet_type"]
        food_footprint = sum(self.food_items[food_type].values())
        activities["food_carbon_footprint"] = food_footprint
        
        # Calculate power consumption and carbon footprint
        activities["total_power_consumption"] = self._calculate_power_consumption(activities)
        activities["carbon_footprint"] = self._calculate_carbon_footprint(activities, food_footprint)
        
        # Add metadata
        activities.update({
            "datetime": date_hour,
            "individual_id": profile["id"],
            "name": profile["name"],
            "role": profile["role"],
            "is_weekend": is_weekend,
            "season": season,
            "hour": hour
        })
        
        return activities

    def _adjust_entertainment_usage(self, profile, hour, gan_output, device_type):
        """Adjust entertainment device usage based on family role and time of day"""
        base_usage = self._scale_value(gan_output, *self.device_patterns["entertainment_devices"][device_type])
        
        # Role-specific adjustments
        if profile["role"] == "Son":
            if device_type in ["smartphone", "laptop"] and (14 <= hour <= 22):
                return base_usage * 1.5  # Increased usage for teenager
        elif profile["role"] == "Daughter":
            if device_type == "tablet" and (15 <= hour <= 19):
                return base_usage * 1.2  # Moderate usage for young child
        elif profile["role"] in ["Grandmother", "Grandfather"]:
            if device_type == "tv_usage" and (8 <= hour <= 22):
                return base_usage * 1.3  # More TV time for elderly
            elif device_type in ["smartphone", "laptop"]:
                return base_usage * 0.4  # Reduced tech usage for elderly
        
        return base_usage

    def _calculate_power_consumption(self, activities):
        """Calculate total power consumption in kWh"""
        # Define power ratings for different devices (in watts)
        power_ratings = {
            "mixer_grinder": 750,
            "pressure_cooker": 1000,
            "microwave": 1200,
            "refrigerator": 200,
            "water_purifier": 25,
            "air_conditioner": 1500,
            "fan": 75,
            "water_heater": 2000,
            "tv": 100,
            "laptop": 65,
            "smartphone_charger": 5,
            "tablet_charger": 10
        }
        
        total_kwh = 0
        
        # Calculate power consumption for each device
        for device, rating in power_ratings.items():
            if f"{device}_usage" in activities:
                usage_hours = activities[f"{device}_usage"] / 60  # Convert minutes to hours
                total_kwh += (rating * usage_hours) / 1000  # Convert watt-hours to kilowatt-hours
        
        # Add power consumption from additional features
        total_kwh += activities["vehicle_monthly_distance_km"] * 0.01  # Example factor
        
        return round(total_kwh, 3)

    def _calculate_carbon_footprint(self, activities, food_footprint):
        """Calculate carbon footprint based on power consumption and food items"""
        # Using India-specific carbon intensity factor (0.82 kg CO2/kWh)
        carbon_footprint = round(activities["total_power_consumption"] * 0.82, 3)
        
        # Add carbon footprint from additional features
        carbon_footprint += activities["vehicle_monthly_distance_km"] * 0.02  # Example factor
        
        # Add carbon footprint from food items
        carbon_footprint += food_footprint
        
        return round(carbon_footprint, 3)

    def generate_time_series(self, days=30, start_date=None):
        """Generate time series data for the entire family"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        all_data = []
        
        for profile in self.profiles:
            for day in range(days):
                current_date = start_date + timedelta(days=day)
                
                for hour in range(24):
                    date_hour = current_date + timedelta(hours=hour)
                    hourly_data = self.generate_hourly_data(profile, date_hour)
                    all_data.append(hourly_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        return df

    def _scale_value(self, gan_output, min_val, max_val):
        """Scale GAN output (-1 to 1) to realistic range"""
        scaled = (gan_output + 1) / 2  # Scale from -1,1 to 0,1
        return min_val + scaled * (max_val - min_val)

    def clean_data(self, df, decimal_places=2):
        """Clean the data by rounding floating point values"""
        float_columns = df.select_dtypes(include=['float64']).columns
        df[float_columns] = df[float_columns].round(decimal_places)
        return df

    def normalize_data(self, df):
        """Normalize the data"""
        scaler = MinMaxScaler()
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        return df

    def save_data(self, df, base_filename="indian_family_lifestyle_data"):
        """Save the generated data and create visualizations"""
        # Clean the data
        df_cleaned = self.clean_data(df)
        
        base_filepath = os.path.join(self.output_dir, base_filename)
        
        # Save main dataset
        df_cleaned.to_csv(f"{base_filepath}.csv", index=False)
        print(f"Data saved to {base_filepath}.csv")
        
        # Save normalized dataset
        df_normalized = self.normalize_data(df_cleaned.copy())
        df_normalized.to_csv(f"{base_filepath}_normalized.csv", index=False)
        print(f"Normalized data saved to {base_filepath}_normalized.csv")
        
        # Create visualizations
        self._create_visualizations(df_cleaned, base_filepath)
        
        # Generate summary statistics
        self._generate_summary(df_cleaned, base_filepath)

    def _create_visualizations(self, df, base_filepath):
        """Create India-specific visualizations"""
        plt.style.use('ggplot')  # Changed from 'seaborn' to 'ggplot'
        
        # 1. Daily power consumption by family member
        plt.figure(figsize=(15, 8))
        for role in df['role'].unique():
            role_df = df[df['role'] == role]
            daily_power = role_df.resample('D', on='datetime')['total_power_consumption'].sum()
            plt.plot(daily_power.index, daily_power.values, label=role)
        
        plt.title('Daily Power Consumption by Family Member')
        plt.xlabel('Date')
        plt.ylabel('Power Consumption (kWh)')
        plt.legend()
        plt.savefig(f"{base_filepath}_daily_power_consumption.png")
        plt.close()
        
        # 2. Carbon footprint by family member
        plt.figure(figsize=(15, 8))
        for role in df['role'].unique():
            role_df = df[df['role'] == role]
            daily_carbon = role_df.resample('D', on='datetime')['carbon_footprint'].sum()
            plt.plot(daily_carbon.index, daily_carbon.values, label=role)
        
        plt.title('Daily Carbon Footprint by Family Member')
        plt.xlabel('Date')
        plt.ylabel('Carbon Footprint (kg CO2)')
        plt.legend()
        plt.savefig(f"{base_filepath}_daily_carbon_footprint.png")
        plt.close()
        
        # 3. Device usage patterns
        device_columns = [col for col in df.columns if 'usage' in col]
        for device in device_columns:
            plt.figure(figsize=(15, 8))
            for role in df['role'].unique():
                role_df = df[df['role'] == role]
                daily_device_usage = role_df.resample('D', on='datetime')[device].sum()
                plt.plot(daily_device_usage.index, daily_device_usage.values, label=role)
            
            plt.title(f'Daily {device.replace("_", " ").title()} by Family Member')
            plt.xlabel('Date')
            plt.ylabel(f'{device.replace("_", " ").title()} (minutes)')
            plt.legend()
            plt.savefig(f"{base_filepath}_{device}_usage.png")
            plt.close()

    def _generate_summary(self, df, base_filepath):
        """Generate summary statistics for the dataset"""
        summary_stats = df.groupby('role').agg({
            'total_power_consumption': ['mean', 'std', 'min', 'max'],
            'carbon_footprint': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        summary_stats.to_csv(f"{base_filepath}_summary_statistics.csv")
        print(f"Summary statistics saved to {base_filepath}_summary_statistics.csv")

    def _get_travel_frequency(self, profile):
        """Get travel frequency based on profile"""
        travel_freq = profile.get("travel_frequency", "never")
        return self.additional_features["vehicle_and_travel"]["frequency_of_traveling_by_air"].get(travel_freq, 0.5)

    def _get_vehicle_type(self, profile):
        """Get vehicle type based on profile"""
        vehicle_type = profile.get("vehicle_type", "petrol")
        return self.additional_features["vehicle_and_travel"]["vehicle_type"].get(vehicle_type, 1.2)

    def _get_waste_bag_size(self, profile):
        """Get waste bag size based on profile"""
        waste_bag_size = profile.get("waste_bag_size", "medium")
        return self.additional_features["day_to_day_activities"]["waste_bag_size"].get(waste_bag_size, 1.0)

def main():
    # Initialize the generator
    generator = IndianFamilyLifestyleGenerator()
    
    # Generate time series data
    print("Generating synthetic time series data...")
    df = generator.generate_time_series(days=30)
    
    # Save the data and create visualizations
    generator.save_data(df)
    
    # Print some basic statistics
    print("\nData Generation Complete!")
    print("\nBasic Statistics:")
    print(df.groupby('role')['carbon_footprint'].describe().round(2))
    
    return df

if __name__ == "__main__":
    df = main()