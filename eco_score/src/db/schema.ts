import {
  pgTable,
  uuid,
  varchar,
  smallserial,
  smallint,
  integer,
  timestamp,
  primaryKey,
} from 'drizzle-orm/pg-core';

export const user = pgTable('user', {
  id: varchar('user_id').primaryKey(),
  firstName: varchar('first_name', { length: 30 }).notNull(),
  lastName: varchar('last_name', { length: 30 }).notNull(),
});

export const userScore = pgTable('user_score', {
  userId: varchar('user_id')
    .notNull()
    .references(() => user.id),
  score: smallint('score').notNull(),
  time: timestamp('timestamp').defaultNow(),
});

export const userCarbonFootprint = pgTable('carbon_footprint', {
  userId: varchar('user_id')
    .notNull()
    .references(() => user.id),
  score: integer('carbon_footprint').notNull(),
  time: timestamp('timestamp').defaultNow(),
});

export const deviceTypes = pgTable('device_types', {
  id: smallserial('id').primaryKey(),
  name: varchar('name').unique().notNull(),
  wattage: integer('wattage').notNull(),
});

export const device = pgTable('device', {
  id: uuid('device_id').primaryKey().defaultRandom(),
  name: varchar('device_name', { length: 20 }).unique().notNull(),
  type: smallserial('device_type_id')
    .notNull()
    .references(() => deviceTypes.id),
});

export const userDevice = pgTable(
  'user_device',
  {
    userId: varchar('user_id')
      .notNull()
      .references(() => user.id),
    deviceId: uuid('device_id')
      .notNull()
      .references(() => device.id),
  },
  (table) => {
    return {
      userDeviceId: primaryKey({
        name: 'user_device_id',
        columns: [table.userId, table.deviceId],
      }),
    };
  }
);

export const deviceReading = pgTable('device_reading', {
  deviceId: uuid('device_id')
    .notNull()
    .references(() => device.id),
  timeOfReading: timestamp('time_of_reading', {
    withTimezone: true,
  }).defaultNow(),
  reading: integer('reading').notNull(),
  timeUsed: timestamp('time_used', { withTimezone: true }),
});

export const activityTypes = pgTable('activity_types', {
  id: smallserial('id').primaryKey(),
  name: varchar('name').unique().notNull(),
});

export const userActivityScore = pgTable('user_activity_score', {
  userId: varchar('user_id')
    .notNull()
    .references(() => user.id),
  activityType: smallserial('activity_type_id')
    .notNull()
    .references(() => activityTypes.id),
  score: smallint('score').notNull(),
  time: timestamp('timestamp', { withTimezone: true }).defaultNow(),
});
