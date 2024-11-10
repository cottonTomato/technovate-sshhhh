import { ReqHandler } from '../../types';
import { IGetDevicesDto, IGetDeviceDataDto, IAddDeviceDto } from './device.dto';
import { db } from '../../db/db';
import { device, userDevice, deviceReading } from '../../db/schema';
import { eq, and, desc } from 'drizzle-orm';
import { StatusCodes } from 'http-status-codes';

type GetDevicesHander = ReqHandler<IGetDevicesDto>;

export const getDevicesHandler: GetDevicesHander = async function (req, res) {
  const userId = req.auth.userId!;

  const devices = await db
    .select({ id: device.id, name: device.name, type: device.type })
    .from(userDevice)
    .innerJoin(device, eq(device.id, userDevice.deviceId))
    .where(eq(userDevice.userId, userId));

  res.status(StatusCodes.OK).json({
    status: 'Success',
    data: devices,
  });
};

type AddDeviceHandler = ReqHandler<IAddDeviceDto>;

export const addDeviceHandler: AddDeviceHandler = async function (req, res) {
  const userId = req.auth.userId!;
  const { name, type } = req.body;

  await db.transaction(async (tx) => {
    const [{ deviceId }] = await tx
      .insert(device)
      .values({
        name,
        type,
      })
      .returning({ deviceId: device.id });

    await tx.insert(userDevice).values({ userId, deviceId });
  });

  res.status(StatusCodes.CREATED).json({
    status: 'Success',
  });
};

type GetDeviceDataHandler = ReqHandler<IGetDeviceDataDto>;

export const getDeviceDataHandler: GetDeviceDataHandler = async function (
  req,
  res
) {
  const userId = req.auth.userId!;
  const { name } = req.body;

  const data = await db
    .select()
    .from(userDevice)
    .innerJoin(device, eq(device.id, userDevice.deviceId))
    .innerJoin(deviceReading, eq(device.id, deviceReading.deviceId))
    .where(and(eq(userDevice.userId, userId), eq(device.name, name)))
    .orderBy(desc(deviceReading.timeOfReading));

  res.status(StatusCodes.OK).json({
    status: 'Success',
    data: data,
  });
};
