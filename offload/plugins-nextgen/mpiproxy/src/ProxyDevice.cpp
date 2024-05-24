#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <tuple>

#include "EventSystem.h"
#include "RemotePluginManager.h"
#include "Shared/APITypes.h"
#include "mpi.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

/// Event Implementations on Device side.
struct ProxyDevice {
  ProxyDevice()
      : NumExecEventHandlers("OMPTARGET_NUM_EXEC_EVENT_HANDLERS", 1),
        NumDataEventHandlers("OMPTARGET_NUM_DATA_EVENT_HANDLERS", 1),
        EventPollingRate("OMPTARGET_EVENT_POLLING_RATE", 1) {
    EventSystem.initialize();
    PluginManager.init();
    for (int PluginId = 0; PluginId < PluginManager.getNumUsedPlugins();
         PluginId++) {
      EventSystem.DevicesPerRemote.emplace_back(
          PluginManager.getNumDevices(PluginId));
    }
  }
  ~ProxyDevice() {
    EventSystem.deinitialize();
    PluginManager.deinit();
  }

  void mapDevicesPerRemote() {
    EventSystem.DevicesPerRemote = {};
    for (int PluginId = 0; PluginId < PluginManager.getNumUsedPlugins();
         PluginId++) {
      EventSystem.DevicesPerRemote.emplace_back(
          PluginManager.getNumDevices(PluginId));
    }
  }

  __tgt_async_info *MapAsyncInfo(void *HostAsyncInfoPtr) {
    __tgt_async_info *TgtAsyncInfoPtr = nullptr;
    if (AsyncInfoTable[HostAsyncInfoPtr])
      TgtAsyncInfoPtr =
          static_cast<__tgt_async_info *>(AsyncInfoTable[HostAsyncInfoPtr]);
    else {
      TgtAsyncInfoPtr = &AsyncInfoList.emplace_back();
      AsyncInfoTable[HostAsyncInfoPtr] = static_cast<void *>(TgtAsyncInfoPtr);
    }

    return TgtAsyncInfoPtr;
  }

  EventTy retrieveNumDevices(MPIRequestManagerTy RequestManager) {
    int32_t NumDevices = PluginManager.getNumDevices();
    RequestManager.send(&NumDevices, 1, MPI_INT);

    co_return (co_await RequestManager);
  }

  EventTy isValidBinary(MPIRequestManagerTy RequestManager) {
    __tgt_device_image Image;
    bool Initialized = false, QueryResult = false;

    uint64_t Size = 0;

    RequestManager.receive(&Size, 1, MPI_UINT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    Image.ImageStart = std::malloc(Size);
    RequestManager.receive(Image.ImageStart, Size, MPI_BYTE);
    RequestManager.receive(&Initialized, sizeof(bool), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    Image.ImageEnd = (void *)((ptrdiff_t)(Image.ImageStart) + Size);

    llvm::SmallVector<std::unique_ptr<GenericPluginTy>> UsedPlugins;

    for (auto &Plugin : PluginManager.Plugins) {
      QueryResult = Plugin->is_valid_binary(&Image, Initialized);
      if (QueryResult) {
        UsedPlugins.emplace_back(std::move(Plugin));
        break;
      }
    }

    for (auto &Plugin : PluginManager.Plugins) {
      if (Plugin)
        UsedPlugins.emplace_back(std::move(Plugin));
    }

    PluginManager.Plugins = std::move(UsedPlugins);
    mapDevicesPerRemote();

    free(Image.ImageStart);
    RequestManager.send(&QueryResult, sizeof(bool), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy initDevice(MPIRequestManagerTy RequestManager) {
    int32_t DeviceId, PluginId;

    RequestManager.receive(&DeviceId, 1, MPI_INT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->init_device(DeviceId);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy initRecordReplay(MPIRequestManagerTy RequestManager) {
    int64_t MemorySize = 0;
    void *VAddr = nullptr;
    bool IsRecord = false, SaveOutput = false;
    uint64_t ReqPtrArgOffset = 0;

    RequestManager.receive(&MemorySize, 1, MPI_INT64_T);
    RequestManager.receive(&VAddr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&IsRecord, sizeof(bool), MPI_BYTE);
    RequestManager.receive(&SaveOutput, sizeof(bool), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t DeviceId, PluginId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->initialize_record_replay(
        DeviceId, MemorySize, VAddr, IsRecord, SaveOutput, ReqPtrArgOffset);

    RequestManager.send(&ReqPtrArgOffset, 1, MPI_UINT64_T);
    co_return (co_await RequestManager);
  }

  EventTy isDataExchangable(MPIRequestManagerTy RequestManager) {
    int32_t DstDeviceId = 0;
    bool QueryResult = false;
    RequestManager.receive(&DstDeviceId, 1, MPI_INT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t SrcDeviceId, PluginId;

    std::tie(PluginId, SrcDeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    QueryResult = PluginManager.Plugins[PluginId]->isDataExchangable(
        SrcDeviceId, DstDeviceId);

    RequestManager.send(&QueryResult, sizeof(bool), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy allocateBuffer(MPIRequestManagerTy RequestManager) {
    int64_t Size = 0;
    int32_t Kind = 0;
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&Kind, 1, MPI_INT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // void *Buffer = malloc(Size);

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    void *Buffer = PluginManager.Plugins[PluginId]->data_alloc(DeviceId, Size,
                                                               nullptr, Kind);

    RequestManager.send(&Buffer, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy deleteBuffer(MPIRequestManagerTy RequestManager) {
    void *Buffer = nullptr;
    int32_t Kind = 0;

    RequestManager.receive(&Buffer, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Kind, 1, MPI_INT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_delete(DeviceId, Buffer, Kind);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy submit(MPIRequestManagerTy RequestManager) {
    void *TgtPtr = nullptr, *HstAsyncInfoPtr = nullptr;
    int64_t Size = 0;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    auto TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    RequestManager.receive(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&TgtPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // TODO: How to wait the submit call on the device?
    void *HstPtr = malloc(Size);
    RequestManager.receiveInBatchs(HstPtr, Size);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    EventDataHandleTy DataHandle(HstPtr, &std::free);
    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_submit_async(
        DeviceId, TgtPtr, DataHandle.get(), Size, nullptr);

    RequestManager.send(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy retrieve(MPIRequestManagerTy RequestManager) {
    void *TgtPtr = nullptr, *HstAsyncInfoPtr = nullptr;
    int64_t Size = 0;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    auto TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    RequestManager.receive(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&TgtPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // TODO: How to wait retrieve call on the device?
    void *HstPtr = malloc(Size);
    EventDataHandleTy DataHandle(HstPtr, &std::free);
    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_retrieve_async(
        DeviceId, DataHandle.get(), TgtPtr, Size, nullptr);

    RequestManager.sendInBatchs(DataHandle.get(), Size);
    RequestManager.send(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy exchange(MPIRequestManagerTy RequestManager) {
    void *SrcPtr = nullptr, *DstPtr = nullptr;
    int DstDeviceId = 0;
    int64_t Size = 0;
    void *HstAsyncInfoPtr = nullptr;

    RequestManager.receive(&SrcPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&DstDeviceId, 1, MPI_INT);
    RequestManager.receive(&DstPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    auto TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    RequestManager.receive(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, SrcDeviceId;

    std::tie(PluginId, SrcDeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_exchange_async(
        SrcDeviceId, SrcPtr, DstDeviceId, DstPtr, Size, nullptr);

    RequestManager.send(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy exchangeSrc(MPIRequestManagerTy RequestManager) {
    void *SrcBuffer;
    int64_t Size;
    int DstDevice;
    // Save head node rank
    int HeadNodeRank = RequestManager.OtherRank;

    RequestManager.receive(&SrcBuffer, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&DstDevice, 1, MPI_INT);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // Set the Destination Rank in RequestManager
    RequestManager.OtherRank = DstDevice;

    // Send buffer to target device
    RequestManager.sendInBatchs(SrcBuffer, Size);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // Set the HeadNode Rank to send the final notificatin
    RequestManager.OtherRank = HeadNodeRank;

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy exchangeDst(MPIRequestManagerTy RequestManager) {
    void *DstBuffer;
    int64_t Size;
    int SrcDevice;
    // Save head node rank
    int HeadNodeRank = RequestManager.OtherRank;

    RequestManager.receive(&DstBuffer, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);
    RequestManager.receive(&SrcDevice, 1, MPI_INT);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // Set the Source Rank in RequestManager
    RequestManager.OtherRank = SrcDevice;

    // Receive buffer from the Source device
    RequestManager.receiveInBatchs(DstBuffer, Size);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // Set the HeadNode Rank to send the final notificatin
    RequestManager.OtherRank = HeadNodeRank;

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy launchKernel(MPIRequestManagerTy RequestManager) {
    void *TgtEntryPtr = nullptr, *HostAsyncInfoPtr = nullptr;
    KernelArgsTy KernelArgs;

    __tgt_async_info *AsyncInfoPtr = nullptr;

    llvm::SmallVector<void *> TgtArgs;
    llvm::SmallVector<ptrdiff_t> TgtOffsets;

    uint32_t NumArgs = 0;

    RequestManager.receive(&NumArgs, 1, MPI_UINT32_T);
    RequestManager.receive(&HostAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    AsyncInfoPtr = MapAsyncInfo(HostAsyncInfoPtr);

    TgtArgs.resize(NumArgs);
    TgtOffsets.resize(NumArgs);

    RequestManager.receive(&TgtEntryPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(TgtArgs.data(), NumArgs * sizeof(void *), MPI_BYTE);
    RequestManager.receive(TgtOffsets.data(), NumArgs * sizeof(ptrdiff_t),
                           MPI_BYTE);

    RequestManager.receive(&KernelArgs, sizeof(KernelArgsTy), MPI_BYTE);
    RequestManager.receive(&AsyncInfoPtr->Queue, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->launch_kernel(
        DeviceId, TgtEntryPtr, TgtArgs.data(), TgtOffsets.data(), &KernelArgs,
        nullptr);

    // Event completion notification
    RequestManager.send(&AsyncInfoPtr->Queue, sizeof(void *), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy loadBinary(MPIRequestManagerTy RequestManager) {
    // Receive the target table sizes.
    size_t ImageSize = 0;
    size_t EntryCount = 0;
    RequestManager.receive(&ImageSize, 1, MPI_UINT64_T);
    RequestManager.receive(&EntryCount, 1, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    llvm::SmallVector<size_t> EntryNameSizes(EntryCount);

    RequestManager.receive(EntryNameSizes.begin(), EntryCount, MPI_UINT64_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    // Create the device name with the appropriate sizes and receive its
    // content.
    // Image = DeviceImage(ImageSize, EntryCount);

    DeviceImage *Image = &RemoteImages.emplace_back(ImageSize, EntryCount);

    Image->setImageEntries(EntryNameSizes);

    // Received the image bytes and the table entries.
    RequestManager.receive(Image->ImageStart, ImageSize, MPI_BYTE);

    for (size_t I = 0; I < EntryCount; I++) {
      RequestManager.receive(&Image->Entries[I].addr, 1, MPI_UINT64_T);
      RequestManager.receive(Image->Entries[I].name, EntryNameSizes[I],
                             MPI_CHAR);
      RequestManager.receive(&Image->Entries[I].size, 1, MPI_UINT64_T);
      RequestManager.receive(&Image->Entries[I].flags, 1, MPI_INT32_T);
      RequestManager.receive(&Image->Entries[I].data, 1, MPI_INT32_T);
    }

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    __tgt_device_binary Binary;

    PluginManager.Plugins[PluginId]->load_binary(DeviceId, Image, &Binary);

    RequestManager.send(&Binary.handle, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy getGlobal(MPIRequestManagerTy RequestManager) {
    __tgt_device_binary Binary;
    uint64_t Size = 0;
    llvm::SmallVector<char> Name;
    void *DevicePtr = nullptr;
    uint32_t NameSize = 0;

    RequestManager.receive(&Binary.handle, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_UINT64_T);
    RequestManager.receive(&NameSize, 1, MPI_UINT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    Name.resize(NameSize);
    RequestManager.receive(Name.data(), NameSize, MPI_CHAR);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->get_global(Binary, Size, Name.data(),
                                                &DevicePtr);

    RequestManager.send(&DevicePtr, sizeof(void *), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy getFunction(MPIRequestManagerTy RequestManager) {
    __tgt_device_binary Binary;
    uint32_t Size = 0;
    llvm::SmallVector<char> Name;
    void *KernelPtr = nullptr;

    RequestManager.receive(&Binary.handle, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_UINT32_T);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    Name.resize(Size);
    RequestManager.receive(Name.data(), Size, MPI_CHAR);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->get_function(Binary, Name.data(),
                                                  &KernelPtr);

    RequestManager.send(&KernelPtr, sizeof(void *), MPI_BYTE);
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy synchronize(MPIRequestManagerTy RequestManager) {
    void *HstAsyncInfoPtr = nullptr;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    auto TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    RequestManager.receive(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->synchronize(DeviceId, TgtAsyncInfo);

    RequestManager.send(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy createEvent(MPIRequestManagerTy RequestManager) {
    void *EventPtr = nullptr;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->create_event(DeviceId, &EventPtr);

    RequestManager.send(&EventPtr, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy recordEvent(MPIRequestManagerTy RequestManager) {
    void *HstAsyncInfoPtr = nullptr, *EventPtr = nullptr;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    auto TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    RequestManager.receive(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&EventPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->record_event(DeviceId, EventPtr,
                                                  TgtAsyncInfo);

    RequestManager.send(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy waitEvent(MPIRequestManagerTy RequestManager) {
    void *HstAsyncInfoPtr = nullptr, *EventPtr = nullptr;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    auto TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);

    RequestManager.receive(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&EventPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->wait_event(DeviceId, EventPtr,
                                                TgtAsyncInfo);

    RequestManager.send(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy syncEvent(MPIRequestManagerTy RequestManager) {
    void *EventPtr = nullptr;
    RequestManager.receive(&EventPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->sync_event(DeviceId, EventPtr);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy destroyEvent(MPIRequestManagerTy RequestManager) {
    void *EventPtr = nullptr;
    RequestManager.receive(&EventPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->destroy_event(DeviceId, EventPtr);

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy initAsyncInfo(MPIRequestManagerTy RequestManager) {
    __tgt_async_info *TgtAsyncInfoPtr = nullptr;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->init_async_info(DeviceId,
                                                     &TgtAsyncInfoPtr);

    RequestManager.send(&TgtAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy initDeviceInfo(MPIRequestManagerTy RequestManager) {
    __tgt_device_info DeviceInfo;
    const char *ErrStr = nullptr;

    RequestManager.receive(&DeviceInfo, sizeof(__tgt_device_info), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->init_device_info(DeviceId, &DeviceInfo,
                                                      &ErrStr);

    RequestManager.send(&DeviceInfo, sizeof(__tgt_device_info), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy queryAsync(MPIRequestManagerTy RequestManager) {
    void *HstAsyncInfoPtr = nullptr;

    RequestManager.receive(&HstAsyncInfoPtr, sizeof(void *), MPI_BYTE);

    if (auto Error = co_await RequestManager; Error)
      co_return Error;

    auto TgtAsyncInfo = MapAsyncInfo(HstAsyncInfoPtr);
    RequestManager.receive(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->query_async(DeviceId, TgtAsyncInfo);

    RequestManager.send(&TgtAsyncInfo->Queue, sizeof(void *), MPI_BYTE);

    co_return (co_await RequestManager);
  }

  EventTy printDeviceInfo(MPIRequestManagerTy RequestManager) {
    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->print_device_info(DeviceId);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataLock(MPIRequestManagerTy RequestManager) {
    void *Ptr = nullptr;
    int64_t Size = 0;
    void *LockedPtr = nullptr;

    RequestManager.receive(&Ptr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_lock(DeviceId, Ptr, Size, &LockedPtr);

    RequestManager.send(&LockedPtr, sizeof(void *), MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataUnlock(MPIRequestManagerTy RequestManager) {
    void *Ptr = nullptr;
    RequestManager.receive(&Ptr, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_unlock(DeviceId, Ptr);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataNotifyMapped(MPIRequestManagerTy RequestManager) {
    void *HstPtr = nullptr;
    int64_t Size = 0;
    RequestManager.receive(&HstPtr, sizeof(void *), MPI_BYTE);
    RequestManager.receive(&Size, 1, MPI_INT64_T);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_notify_mapped(DeviceId, HstPtr, Size);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy dataNotifyUnmapped(MPIRequestManagerTy RequestManager) {
    void *HstPtr = nullptr;
    RequestManager.receive(&HstPtr, sizeof(void *), MPI_BYTE);

    if (auto Err = co_await RequestManager; Err)
      co_return Err;

    int32_t PluginId, DeviceId;

    std::tie(PluginId, DeviceId) =
        EventSystem.mapDeviceId(RequestManager.DeviceId);

    PluginManager.Plugins[PluginId]->data_notify_unmapped(DeviceId, HstPtr);

    RequestManager.send(nullptr, 0, MPI_BYTE);
    co_return (co_await RequestManager);
  }

  EventTy exit(MPIRequestManagerTy RequestManager,
               std::atomic<EventSystemStateTy> &EventSystemState) {
    EventSystemStateTy OldState =
        EventSystemState.exchange(EventSystemStateTy::EXITED);
    assert(OldState != EventSystemStateTy::EXITED &&
           "Exit event received multiple times");

    // Event completion notification
    RequestManager.send(nullptr, 0, MPI_BYTE);

    co_return (co_await RequestManager);
  }

  /// Function executed by the event handler threads.
  void runEventHandler(std::stop_token Stop, EventQueue &Queue) {
    while (EventSystem.EventSystemState == EventSystemStateTy::RUNNING ||
           Queue.size() > 0) {
      EventTy Event = Queue.pop(Stop);

      // Re-checks the stop condition when no event was found.
      if (Event.empty()) {
        continue;
      }

      Event.resume();

      if (!Event.done()) {
        Queue.push(std::move(Event));
      }

      auto Error = Event.getError();
      if (Error)
        REPORT("Internal event failed with msg: %s\n",
               toString(std::move(Error)).data());
    }
  }

  /// Gate thread procedure.
  ///
  /// Caller thread will spawn the event handlers, execute the gate logic and
  /// wait until the event system receive an Exit event.
  void runGateThread() {
    // Device image to be used by this gate thread.
    // DeviceImage Image;

    // Updates the event state and
    EventSystem.EventSystemState = EventSystemStateTy::RUNNING;

    // Spawns the event handlers.
    llvm::SmallVector<std::jthread> EventHandlers;
    EventHandlers.resize(NumExecEventHandlers.get() +
                         NumDataEventHandlers.get());
    int EventHandlersSize = EventHandlers.size();
    auto HandlerFunction = std::bind_front(&ProxyDevice::runEventHandler, this);
    for (int Idx = 0; Idx < EventHandlersSize; Idx++) {
      EventHandlers[Idx] = std::jthread(
          HandlerFunction, std::ref(Idx < NumExecEventHandlers.get()
                                        ? EventSystem.ExecEventQueue
                                        : EventSystem.DataEventQueue));
    }

    // Executes the gate thread logic
    while (EventSystem.EventSystemState == EventSystemStateTy::RUNNING) {
      // Checks for new incoming event requests.
      MPI_Message EventReqMsg;
      MPI_Status EventStatus;
      int HasReceived = false;
      MPI_Improbe(MPI_ANY_SOURCE,
                  static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                  EventSystem.GateThreadComm, &HasReceived, &EventReqMsg,
                  MPI_STATUS_IGNORE);

      // If none was received, wait for `EVENT_POLLING_RATE`us for the next
      // check.
      if (!HasReceived) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(EventPollingRate.get()));
        continue;
      }

      // Acquires the event information from the received request, which are:
      // - Event type
      // - Event tag
      // - Target comm
      // - Event source rank
      int EventInfo[3];
      MPI_Mrecv(EventInfo, 3, MPI_INT, &EventReqMsg, &EventStatus);
      const auto NewEventType = static_cast<EventTypeTy>(EventInfo[0]);
      MPIRequestManagerTy RequestManager(
          EventSystem.getNewEventComm(EventInfo[1]), EventInfo[1],
          EventStatus.MPI_SOURCE, EventInfo[2]);

      // Creates a new receive event of 'event_type' type.
      using enum EventTypeTy;
      EventTy NewEvent;
      switch (NewEventType) {
      case RETRIEVE_NUM_DEVICES:
        NewEvent = retrieveNumDevices(std::move(RequestManager));
        break;
      case IS_VALID_BINARY:
        NewEvent = isValidBinary(std::move(RequestManager));
        break;
      case INIT_DEVICE:
        NewEvent = initDevice(std::move(RequestManager));
        break;
      case INIT_RECORD_REPLAY:
        NewEvent = initRecordReplay(std::move(RequestManager));
        break;
      case IS_DATA_EXCHANGABLE:
        NewEvent = isDataExchangable(std::move(RequestManager));
        break;
      case ALLOC:
        NewEvent = allocateBuffer(std::move(RequestManager));
        break;
      case DELETE:
        NewEvent = deleteBuffer(std::move(RequestManager));
        break;
      case SUBMIT:
        NewEvent = submit(std::move(RequestManager));
        break;
      case RETRIEVE:
        NewEvent = retrieve(std::move(RequestManager));
        break;
      case LOCAL_EXCHANGE:
        NewEvent = exchange(std::move(RequestManager));
        break;
      case EXCHANGE_SRC:
        NewEvent = exchangeSrc(std::move(RequestManager));
        break;
      case EXCHANGE_DST:
        NewEvent = exchangeDst(std::move(RequestManager));
        break;
      case EXIT:
        NewEvent =
            exit(std::move(RequestManager), EventSystem.EventSystemState);
        break;
      case LOAD_BINARY:
        NewEvent = loadBinary(std::move(RequestManager));
        break;
      case GET_GLOBAL:
        NewEvent = getGlobal(std::move(RequestManager));
        break;
      case GET_FUNCTION:
        NewEvent = getFunction(std::move(RequestManager));
        break;
      case LAUNCH_KERNEL:
        NewEvent = launchKernel(std::move(RequestManager));
        break;
      case SYNCHRONIZE:
        NewEvent = synchronize(std::move(RequestManager));
        break;
      case CREATE_EVENT:
        NewEvent = createEvent(std::move(RequestManager));
        break;
      case RECORD_EVENT:
        NewEvent = recordEvent(std::move(RequestManager));
        break;
      case WAIT_EVENT:
        NewEvent = waitEvent(std::move(RequestManager));
        break;
      case SYNC_EVENT:
        NewEvent = syncEvent(std::move(RequestManager));
        break;
      case DESTROY_EVENT:
        NewEvent = destroyEvent(std::move(RequestManager));
        break;
      case INIT_ASYNC_INFO:
        NewEvent = initAsyncInfo(std::move(RequestManager));
        break;
      case INIT_DEVICE_INFO:
        NewEvent = initDeviceInfo(std::move(RequestManager));
        break;
      case QUERY_ASYNC:
        NewEvent = queryAsync(std::move(RequestManager));
        break;
      case PRINT_DEVICE_INFO:
        NewEvent = printDeviceInfo(std::move(RequestManager));
        break;
      case DATA_LOCK:
        NewEvent = dataLock(std::move(RequestManager));
        break;
      case DATA_UNLOCK:
        NewEvent = dataUnlock(std::move(RequestManager));
        break;
      case DATA_NOTIFY_MAPPED:
        NewEvent = dataNotifyMapped(std::move(RequestManager));
        break;
      case DATA_NOTIFY_UNMAPPED:
        NewEvent = dataNotifyUnmapped(std::move(RequestManager));
        break;
      case SYNC:
      case EXCHANGE:
        assert(false && "Trying to create a local event on a remote node");
      }

      if (NewEventType == LAUNCH_KERNEL) {
        EventSystem.ExecEventQueue.push(std::move(NewEvent));
      } else {
        EventSystem.DataEventQueue.push(std::move(NewEvent));
      }
    }

    assert(EventSystem.EventSystemState == EventSystemStateTy::EXITED &&
           "Event State should be EXITED after receiving an Exit event");
  }

private:
  llvm::SmallVector<__tgt_async_info> AsyncInfoList{};
  llvm::SmallVector<DeviceImage, 1> RemoteImages;
  llvm::DenseMap<void *, void *> AsyncInfoTable;
  RemotePluginManager PluginManager;
  EventSystemTy EventSystem;
  /// Number of execute event handlers to spawn.
  IntEnvar NumExecEventHandlers;
  /// Number of data event handlers to spawn.
  IntEnvar NumDataEventHandlers;
  /// Polling rate period (us) used by event handlers.
  IntEnvar EventPollingRate;
};

int main(int argc, char **argv) {
  ProxyDevice PD;
  PD.runGateThread();
  return 0;
}